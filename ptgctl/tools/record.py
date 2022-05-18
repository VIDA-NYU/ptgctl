'''Record data from the API to file. PARTIALLY DEVELOPED / UNTESTED


'''
import os
import io
import asyncio
from .. import util

__bind__ = ['store_streams']


class Disk:
    def __init__(self, path='./data'):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def store_entries(self, entries, stream_id=None):
        fn, archive = zip_entries(entries, stream_id)
        with open(os.path.join(self.directory, fn), 'wb') as f:
            f.write(archive)


def zip_entries(entries, stream_id=None):
    import zipfile
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED, False) as zf:
        for ts, data in entries:
            zf.writestr(ts.decode('utf-8'), data[b'd'])
    prefix = f'{stream_id}_' if stream_id else ''
    fn = f'{prefix}{entry2time(entries[0])}_{entry2time(entries[-1])}.zip'
    return fn, archive.getvalue()


def entry2time(entry):
    return entry[0].decode('utf-8').split('-')[0]



async def api_reader(api, stream_id, delay=1, raw_holo=False, **kw):
    '''Show a video stream from the API.'''
    async with api.data_pull_connect(stream_id, **kw) as ws:
        while True:
            for sid, ts, data in await ws.aread():
                yield sid, ts, data

async def as_batches(it, max_size=9500000, max_len=1000):
    while True:
        size = 0
        entries = []
        for sid, ts, data in it:
            size += len(data)
            entries.extend([ts, data])
            if len(entries) > max_len or size > max_size:
                break
        else:
            break
        if entries:
            yield entries


WRITERS = {
    'disk': Disk,
}


async def store_data_stream(api, stream_id, writer='disk'):
    drive = WRITERS[writer]
    async for entries in as_batches(api_reader(api, stream_id)):
        if entries:
            drive.store_entries(entries, stream_id)


@util.async2sync
async def store_streams(api, *stream_ids, **kw):
    return await asyncio.gather(*(
        store_data_stream(api, sid, **kw) for sid in stream_ids))

