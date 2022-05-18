import io
import asyncio
import websockets

import numpy as np
from PIL import Image
import cv2


async def webcam_loop(api, src=0, **kw):
    sid = 'main'
    await asyncio.gather(
        webcam(api, sid, src, **kw),
        imshow(api, sid),
    )

async def imshow(api, stream_id, delay=1, raw_holo=False, **kw):
    '''Show a video stream from the API.'''
    import cv2
    async with api.data_pull_connect(stream_id, **kw) as ws:
        print('imshow connected')
        while True:
            for sid, ts, data in await ws.aread():
                im = np.array(Image.open(io.BytesIO(data)))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                print('pull', im.shape)
                cv2.imshow(sid, im)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break


async def webcam(api, sid, src=0):
    async with api.data_push_connect(sid) as ws:
        for im in _webcam_feed(src):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            print('write', im.shape)
            await ws.awrite(_img_dump(im))

def _webcam_feed(src):
    cap = cv2.VideoCapture(src)
    while True:
        ret, im = cap.read()
        if not ret:
            break
        yield im

def _img_dump(im, format='jpeg'):
    output = io.BytesIO()
    Image.fromarray(im).save(output, format=format)
    return output.getvalue()


# def main(api, sid='main', src=0, **kw):
#     return asyncio.run(asyncio.gather(
#         webcam(api, sid, src, **kw),
#         imshow(api, sid),
#     ))

WSPULL = 'ws://localhost:7890/data/main/pull'
WSPUSH = 'ws://localhost:7890/data/main/push'
TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiZWEiLCJleHAiOjE2NTUyMTc0NDN9.llgTgFoAHt_zzNKXmGPFov24Wg3F4aHkCVcnd4wdia0'

async def _main():
    print('hi', 0)
    W = websockets.connect(WSPULL, extra_headers={'Authorization': f'Bearer {TOKEN}'})
    print(type(W))
    async with W as ws1:
        print(type(ws1))
        print('hi', 1, ws1)
        async with websockets.connect(WSPUSH, extra_headers={'Authorization': f'Bearer {TOKEN}'}) as ws2:
            print('hi', 2, ws1, ws2)
        print('hi', 3, ws1)
    print('hi', 4)

async def _maini():
    print('hi', 0)
    W = websockets.connect(WSPULL, extra_headers={'Authorization': f'Bearer {TOKEN}'})
    async with W as ws1:
        await asyncio.sleep(1)

async def _main():
    return await asyncio.gather(_maini(), _maini(), _maini())

def main():
    return asyncio.run(_main())

if __name__ == '__main__':
    import fire
    fire.Fire(main)