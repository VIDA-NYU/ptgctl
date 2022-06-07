'''Various tools for faking hololens data.


WARNING: this doesn't upload video in the right format. 

'''
import io
import asyncio
import cv2
from PIL import Image as pil

from ptgctl.tools.audio import AudioRecorder
from .. import util



CAM_POS_SIDS = ['main', 'glr', 'glf', 'grf', 'grr']


async def _imshow(api, sid, *a, **kw):
    from . import display
    return await display.imshow.asyncio(api, sid, *a, **kw)

@util.async2sync
async def video_loop(api, src=0, pos=0, **kw):    
    '''Send video (by default your webcam) to the API and pull it back to display on screen.'''
    sid = CAM_POS_SIDS[pos]
    await util.async_first_done(
        video.asyncio(api, src, pos, **kw),
        _imshow(api, sid),
    )


DIVS = 4
@util.async2sync
async def video(api, src=0, pos=0, width=0.3):
    '''Send video (by default your webcam) to the API.'''
    sid = CAM_POS_SIDS[pos]
    async with api.data_push_connect(sid) as ws:
        for im in _video_feed(src):
            if pos:
                im = _fake_side_cam(im, pos, width)
            else:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            await ws.send_data(_img_dump(im))

def _img_dump(im, format='jpeg'):
    output = io.BytesIO()
    pil.fromarray(im).save(output, format=format)
    return output.getvalue()

def _fake_side_cam(im, pos=0, width=0.3):
    if pos:  # emulate l/r grey cameras
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, W = im.shape[:2]
        i = int(W * (1 - width) * (pos - 1) / (DIVS - 1))
        im = im[:, i:i + int(W * width)]
    return im


def _video_feed(src):
    cap = cv2.VideoCapture(src)
    while True:
        ret, im = cap.read()
        if not ret:
            break
        yield im



@util.async2sync
async def audio(api, sid='mic0', device=None):
    from .audio import AudioRecorder, pack_audio
    with AudioRecorder(device=device) as rec:
        async with api.data_push_connect(sid) as ws:
            while True:
                y, pos = rec.read()
                if y is None:
                    await asyncio.sleep(rec.block_duration / 5)
                    continue
                print(y.shape, pos)
                await ws.send_data(pack_audio(y, pos, rec.sr, rec.channels))

        

