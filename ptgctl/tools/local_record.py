'''Record data from the API to file. PARTIALLY DEVELOPED / UNTESTED


'''
import os
import io
import glob
import time
import asyncio
from .. import util
import tqdm

# __bind__ = ['store']

import ptgctl
ptgctl.log.setLevel('WARNING')


def tqprint(*a, **kw):
    tqdm.tqdm.write(' '.join(map(str, a)), **kw)


class Disk:
    EXT = '.zip'
    def __init__(self, path='./data'):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def list(self, stream_id):
        return sorted(glob.glob(os.path.join(self.path, stream_id, f'**/*{self.EXT or ""}')))

    def store(self, entries, stream_id):
        fname, archive = _zip(entries)
        fname = os.path.join(self.path, stream_id, fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as f:
            f.write(archive)
        tqprint(
            'wrote', fname, len(entries), _pretty_bytes(len(archive)), 
            util.parse_time(entries[0][0]).strftime('%X.%f'), 
            util.parse_time(entries[-1][0]).strftime('%X.%f'))

    def load(self, fname):
        with open(fname, 'rb') as f:
            for ts, data in _unzip(f.read()):
                yield ts, data


def _pretty_bytes(b, scale=1000, names=['b', 'kb', 'mb', 'gb', 'tb']):
    return next((
            f'{b / (scale**i):.1f}{n}' 
            for i, n in enumerate(names) 
            if b / (scale**(i+1)) < 1
        ), 
        f'{b / (scale**(len(names)-1))}{names[-1]}')


WRITERS = {
    'disk': Disk,
}

def get_writer(name, *a, **kw):
    return WRITERS[name](*a, **kw)


def _zip(entries):
    import zipfile
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED, False) as zf:
        for ts, data in entries:
            zf.writestr(ts, data)
    date = util.parse_time(entries[0][0]).strftime('%Y-%m-%d')
    fn = f'{date}/{entries[0][0]}_{entries[-1][0]}.zip'
    return fn, archive.getvalue()


def _unzip(data):
    import zipfile
    archive = io.BytesIO(data)
    with zipfile.ZipFile(archive, 'r', zipfile.ZIP_STORED, False) as zf:
        for ts in sorted(zf.namelist()):
            with zf.open(ts, 'r') as f:
                data = f.read()
                yield ts, data


# async def _api_streamer(api, stream_id, **kw):
#     '''Show a video stream from the API.'''
#     async with api.data_pull_connect(stream_id, **kw) as ws:
#         while True:
#             data = await ws.recv_data()
#             if not data:
#                 return
#             for sid, ts, data in data:
#                 print('data')
#                 yield sid, ts, data


async def _api_reader(api, stream_id, last_entry_id=0, **kw):
    '''Show a video stream from the API.'''
    # stream_info = api.streams.get(stream_id)
    # final_id = stream_info['info']['last-entry']

    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await loop.run_in_executor(None, lambda: api.data(stream_id, last_entry_id=last_entry_id, timeout=11))
            if not data:
                return
            for sid, ts, d in data: #api.data(stream_id, last_entry_id=last_entry_id):
                if ts == last_entry_id:
                    return
                yield sid, ts, d
                last_entry_id = ts
    except KeyboardInterrupt:
        print('Interrupted')



async def _as_batches(it, max_size=29500000, max_len=2000):
    while True:
        size = 0
        entries = []
        stream_id = None
        try:
            with tqdm.tqdm(total=max_len) as pbar:
                async for sid, ts, data in it:
                    stream_id = stream_id or sid
                    size += len(data)
                    entries.append((ts, data))

                    pbar.update()
                    pbar.set_description(f'{sid} {util.parse_time(ts).strftime("%x %X.%f")} {ts} size={size}/{max_size} len={len(entries)}/{max_len}')
                    if len(entries) >= max_len or size > max_size:
                        break
                else:
                    break
        finally:
            if entries:
                yield stream_id, entries


async def _store_stream(api, stream_id, writer='disk', resume=False, **kw):
    drive = WRITERS[writer]()

    if 'last_entry_id' not in kw and resume:
        last_entry_id = None
        for last_entry_id, data in drive.load(drive.list(stream_id)[-1]):
            pass
        if last_entry_id:
            tqprint(stream_id, 'resuming at', last_entry_id, util.parse_time(last_entry_id).strftime('%c'))
            kw['last_entry_id'] = last_entry_id

    with tqdm.tqdm() as pbar:
        async for stream_id, entries in _as_batches(_api_reader(api, stream_id, **kw)):
            if stream_id and entries:
                drive.store(entries, stream_id)
                pbar.update(len(entries))


@util.async2sync
async def store(api, *stream_ids, **kw):
    if any(s == '*' for s in stream_ids):
        stream_ids = api.streams.ls()
        print('Using all Stream IDs:', stream_ids)
    # make sure it's one stream per file
    stream_ids = (s for ss in stream_ids for s in ss.split('+'))
    await asyncio.gather(*(_store_stream(api, sid, **kw) for sid in stream_ids))



async def _replay_stream(api, stream_id, writer='disk', fullspeed=False):
    drive = WRITERS[writer]()

    async with api.data_push_connect(stream_id) as ws:

        t_last = None
        t0_last = time.time()
        with tqdm.tqdm() as pbar:

            for fname in drive.list():
                for ts, data in drive.load(fname):
                    await ws.send_data(data)
                    pbar.update()

                    if not fullspeed:
                        t = util.parse_epoch_time(ts)
                        t_last = t_last or t
                        time.sleep(max(0, (t - t_last) - (time.time() - t0_last)))
                        t0_last = time.time()

@util.async2sync
async def replay(api, *stream_ids, **kw):
    stream_ids = (s for ss in stream_ids for s in ss.split('+'))
    return await asyncio.gather(*(_replay_stream(api, sid, **kw) for sid in stream_ids))


@util.async2sync
async def summary(api, *stream_ids, writer='disk', **kw):
    drive = WRITERS[writer]()
    stream_ids = stream_ids or api.streams.ls()

    for sid in stream_ids:
        info = api.streams.get(sid)
        print(sid)
        print(info['meta'])
        print(info['info'])
        fs = drive.list(sid)
        count = sum(1 for f in fs for _ in drive.load(f))
        print(f'saved {count}/{info["info"]["length"]} entries.')
        print('')