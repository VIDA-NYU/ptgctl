import time
import ptgctl


@ptgctl.util.async2sync
async def recrepl():
    api = ptgctl.API()

    async with api.data_push_connect(['a'], recording_name='asdf') as ws_push:
            for i in range(100):
                await ws_push.send_data([f'hi {i}'.encode()], 'a')
                time.sleep(0.001)
            await ws_push.send_data(['bye'.encode()], 'a')

    timestamps = []
    async with api.data_pull_connect(['a'], recording_name='asdf') as ws_pull, \
               api.data_push_connect(['b'], recording_name='asdf') as ws_push:
            for i in range(100):
                for sid, timestamp, data in await ws_pull.recv_data():
                    print(sid, timestamp, data)
                    timestamps.append(timestamp)
                    if data == b'bye': break
                    assert data.startswith(b'hi')
                    await ws_push.send_data(['ok'.encode()], 'b', timestamp)

    timestamps2 = []
    async with api.data_pull_connect(['b'], recording_name='asdf') as ws_pull:
            for i in range(100):
                for sid, timestamp, data in await ws_pull.recv_data():
                    print(sid, timestamp, data)
                    timestamps2.append(timestamp)
                    if data == b'bye': break
                    assert data == b'ok'

    assert timestamps == timestamps2

def test_record_replay():
    recrepl()