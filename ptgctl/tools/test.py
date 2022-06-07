import logging
import ptgctl
import tqdm
from .. import holoframe
from .. import util

ptgctl.core.log.setLevel(logging.ERROR)

def _fps(api, stream_id, **kw):
    with tqdm.tqdm() as pbar:
        while True:
            for sid, ts, data_bytes in api.data(stream_id, **kw):
                data = holoframe.load(data_bytes)
                pbar.update()
                pbar.set_description(f'{sid} {util.parse_time(ts).strftime("%c")} keys={set(data)}')



@util.async2sync
async def _fps_stream(api, stream_id, **kw):
    async with api.data_pull_connect(stream_id, **kw) as ws:
        with tqdm.tqdm() as pbar:
            while True:
                for sid, ts, data_bytes in await ws.recv_data():
                    data = holoframe.load(data_bytes)
                    pbar.update()
                    pbar.set_description(f'{sid} {util.parse_time(ts).strftime("%c")} keys={set(data)}')



def fps(api, *a, stream=False, **kw):
    if stream:
        _fps_stream(api, *a, **kw)
    else:
        _fps(api, *a, **kw)