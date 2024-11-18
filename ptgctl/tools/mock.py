'''Various tools for faking hololens data.


WARNING: this doesn't upload video in the right format. 

'''
import os
import io
import time
import asyncio
import cv2
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
async def video(api, src=0, pos=0, width=0.3, shape=None, fps=15, speed=1, stepbystep=False, prefix=None, skill=None):
    '''Send video (by default your webcam) to the API.'''
    sid = CAM_POS_SIDS[pos]
    sid = f'{prefix or ""}{sid}'
    if skill:
        api.session.start_recipe(skill)
    tlast = 0
    async with api.data_push_connect(sid, batch=True) as ws:
        async for im in _video_feed(src, fps, shape, speed=speed):
            if pos:
                im = _fake_side_cam(im, pos, width)
            # else:
            #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # print(dump_v3(im))
            # print(load(dump_v3(im)))
            if stepbystep:
                input()
            t=time.time()
            await ws.send_data([dump_v3(im)], [sid], [util.format_epoch_time(t, tlast)])
            tlast = t
    
def _img_dump(im, format='jpeg'):
    from PIL import Image
    output = io.BytesIO()
    Image.fromarray(im).save(output, format=format)
    return output.getvalue()



def _fake_side_cam(im, pos=0, width=0.3):
    if pos:  # emulate l/r grey cameras
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, W = im.shape[:2]
        i = int(W * (1 - width) * (pos - 1) / (DIVS - 1))
        im = im[:, i:i + int(W * width)]
    return im


async def _video_feed(src, fps=None, shape=None, speed=None):
    import tqdm
    if src is False:
        import numpy as np
        if isinstance(shape, int):
            shape = (shape, shape, 3)
        if shape is None:
            shape = (300, 400, 3)
        with tqdm.tqdm() as pbar:
            while True:
                yield np.random.uniform(0, 255, shape).astype('uint8')
                pbar.update()
                if fps:
                    await asyncio.sleep(1/fps)
            return
    cap = cv2.VideoCapture(src)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = n_frames/vid_fps
    fps = fps or vid_fps
    skip = (vid_fps / fps if fps else 1)*speed
    if not cap.isOpened():
        raise ValueError(f"{cap}")
    
    t0 = t00 = time.time()
    lag = 0
    fps_counter = 0
    with tqdm.tqdm(total=int(duration*fps/speed), desc=f'{os.path.basename(src)}: fps={vid_fps:.0f}->{fps:.0f} [x{speed:.1g}]. duration={duration:.1f}s') as pbar:
        while True:
            ret, im = cap.read()
            if not ret:
                break
            fps_counter += 1
            while fps_counter >= skip:
                fps_counter -= skip

                if isinstance(shape, int):
                    ratio = max(shape/im.shape[0], shape/im.shape[1])
                    im = cv2.resize(im, (0, 0), fx=ratio, fy=ratio)
                elif shape:
                    im = cv2.resize(im, (shape[1], shape[0]))
                yield im
                pbar.update()
                if fps:
                    t = time.time()
                    dt = 1/fps - (t-t0) - lag
                    await asyncio.sleep(max(0, dt))
                    lag = max(-dt, 0)
                    t0 = t + max(0, dt)
    print(f"Done. took {time.time()-t00:.1f} seconds.")


@util.async2sync
async def audio(api, sid='mic0', device=None, weird_offset=1.6):
    from .audio import AudioRecorder, pack_audio
    with AudioRecorder(device=device) as rec:
        offset = 0
        while True:
            async with api.data_push_connect(sid) as ws:
                while True:
                    y, pos = rec.read()
                    if y is None:
                        await asyncio.sleep(1e-6)
                        continue
                    print(y.shape, pos, pos-offset, len(ws.ws.messages), rec.q.qsize())
                    await ws.send_data(
                        [pack_audio(y, pos, rec.sr, rec.channels)],
                        [sid],
                    )

                    # for some reason, after appoximately this many samples,
                    # the websocket stops streaming and will stop streaming 
                    # until around 2.2e6 at which point it crashes. To fix,
                    # we just restart the websocket before it gets to that 
                    # point.
                    if weird_offset and pos-offset > weird_offset * 1e6:
                        offset=pos
                        break

def wsvars(ws):
    return {(k, v) for k, v in vars(ws).items() if isinstance(v, (int, str, bool))}


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

