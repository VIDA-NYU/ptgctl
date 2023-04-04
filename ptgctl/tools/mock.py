'''Various tools for faking hololens data.


WARNING: this doesn't upload video in the right format. 

'''
import io
import time
import asyncio
import cv2
from PIL import Image as pil

from ptgctl.tools.audio import AudioRecorder
from .. import util
from ptgctl.holoframe import dump_v3, load


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
async def video(api, src=0, pos=0, width=0.3, shape=(300, 400, 3), fps=30, stepbystep=False, prefix=None):
    '''Send video (by default your webcam) to the API.'''
    sid = CAM_POS_SIDS[pos]
    sid = f'{prefix or ""}{sid}'
    async with api.data_push_connect(sid, batch=True) as ws:
        for im in _video_feed(src, shape):
            if pos:
                im = _fake_side_cam(im, pos, width)
            # else:
            #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # print(dump_v3(im))
            # print(load(dump_v3(im)))
            if stepbystep:
                input()
            await ws.send_data([dump_v3(im)], [sid], [util.format_epoch_time(time.time())])
            await asyncio.sleep(1/fps)

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


def _video_feed(src, shape=(300, 400, 3)):
    import tqdm
    with tqdm.tqdm() as pbar:
        if src is False:
            import numpy as np
            while True:
                yield np.random.uniform(0, 255, shape).astype('uint8')
                pbar.update()
            return
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise ValueError(f"{cap}")
        while True:
            ret, im = cap.read()
            if not ret:
                break
            yield im
            pbar.update()

@util.async2sync
async def audio(api, sid='mic0', device=None):
    from .audio import AudioRecorder, pack_audio
    with AudioRecorder(device=device) as rec:
        async with api.data_push_connect(sid) as ws:
            while True:
                y, pos = rec.read()
                if y is None:
                    await asyncio.sleep(1e-6)
                    continue
                print(y.shape, pos)
                await ws.send_data(
                    [pack_audio(y, pos, rec.sr, rec.channels)],
                    [sid],
                )

        

@util.async2sync
async def movie(api, src, fps=15):
    import time
    import numpy as np
    import ptgctl
    import ptgctl.util
    from ptgctl import holoframe
    from ptgctl.tools.audio import AudioPlayer, pack_audio

    from moviepy.video.io.VideoFileClip import VideoFileClip

    # api = ptgctl.API('test', 'test')
    sids = ['main', 'mic0'] # 

    start_time = time.time()
    pos = 0

    video = VideoFileClip(src)
    audio = video.audio
    audio_data = audio.to_soundarray().astype(np.float32)
    audio_sr = audio.fps

    # with AudioPlayer() as player:
    async with api.data_push_connect(sids) as ws:
        for t in np.arange(0, video.duration, 1/fps):
            
            video_frame = video.get_frame(t)
            audio_frame = audio_data[int(t * audio_sr):int((t + 1/fps) * audio_sr)][:,:1]#.mean(axis=1, keepdims=True)

            # pos += audio_frame.size
            pos = round((t + 1/fps) * audio_sr)
            await ws.send_data(
                [
                    holoframe.dump_v3(video_frame[:,:,::-1]), # [H, W, C]: BGR
                    pack_audio(audio_frame, pos, audio_sr, 1), # [time, 1]
                ], 
                sids,
                [ptgctl.util.format_epoch_time(start_time + t)]*len(sids))
            await asyncio.sleep(1/fps)

