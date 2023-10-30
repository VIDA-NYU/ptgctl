import time
import ptgctl


@ptgctl.util.async2sync
async def recrepl():
    api = ptgctl.API()

    async with api.data_push_connect(['a'], recording_name='asdf') as ws_push:
            for i in range(10):
                await ws_push.send_data([f'"hi {i}"'.encode()], 'a')
                time.sleep(0.001)
            await ws_push.send_data(['"bye"'.encode()], 'a')

    timestamps = []
    async with api.data_pull_connect(['a'], recording_name='asdf') as ws_pull, \
               api.data_push_connect(['b'], recording_name='asdf', write_json=True) as ws_push:
            for i in range(10):
                for sid, timestamp, data in await ws_pull.recv_data():
                    print(sid, timestamp, data)
                    if data == b'"bye"': break
                    timestamps.append(timestamp)
                    assert data.startswith(b'"hi')
                    await ws_push.send_data(['"ok"'.encode()], 'b', timestamp)
                await ws_push.send_data(['"bye"'.encode()], 'a')

    timestamps2 = []
    async with api.data_pull_connect(['b'], recording_name='asdf') as ws_pull:
            for i in range(10):
                for sid, timestamp, data in await ws_pull.recv_data():
                    print(sid, timestamp, data)
                    if data == b'"bye"': break
                    timestamps2.append(timestamp)
                    assert data == b'"ok"'

    assert timestamps == timestamps2

def test_record_replay():
    recrepl()