'''This contains functions to display values from the API in interesting ways.

'''
import io
import time
import asyncio
import datetime
import numpy as np
from PIL import Image
from .. import util

log = util.getLogger(__name__, 'debug')


@util.async2sync
async def imshow(api, stream_id, delay=1, raw_holo=False, **kw):
    '''Show a video stream from the API.'''
    import cv2
    if raw_holo:
        from .. import holoframe
    async with api.data_pull_connect(stream_id, **kw) as ws:
        t0 = time.time()
        i = 0
        while True:
            for sid, ts, data in await ws.recv_data():
                i += 1
                if raw_holo:
                    im = holoframe.load(data)['image']
                else:
                    im = np.array(Image.open(io.BytesIO(data)))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                ts_frame = util.ts2datetime(ts)
                latency = datetime.datetime.now() - ts_frame
                text = f"{ts_frame.strftime('%c.%f')} [fps={i / (time.time() - t0):.1f}, {latency} latency]"
                cv2.putText(im, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0))
                cv2.imshow(sid, im)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break


def imshow1(api, stream_id, raw_holo=False, **kw):
    '''Show a single frame of a stream.'''
    import cv2
    if raw_holo:
        from .. import holoframe
    for sid, ts, data in api.data(stream_id, **kw):
        if raw_holo:
            im = holoframe.load(data)['image']
        else:
            im = np.array(Image.open(io.BytesIO(data)))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow(f'{sid}:{ts}', im)
    cv2.waitKey(0)


def local_video(api=None, src=0, pos=0, width=0.3, fps=40):
    '''Capture your webcam and display on screen. This does not interact with the API.
    
    Arguments:
        api (ptgctl.API): for compatability reasons. Not used.
        src (int, str): The video source to be used.
        pos (int): The camera position to emulate. 
    '''
    import cv2
    from .mock import CAM_POS_SIDS, _video_feed, _fake_side_cam
    sid = CAM_POS_SIDS[pos]
    delay = int(1. / fps * 1000) or 1 if fps else 0
    for im in _video_feed(src):
        if pos:
            im = _fake_side_cam(im, pos, width)
        cv2.imshow(f'webcam:{src}:{sid}', im)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break



def test(api, stream_id=None, **kw):
    if not stream_id:
        import json
        return json.dumps(api.streams(), indent=4)
    for sid, ts, data in api.data(stream_id, **kw):
        print(sid, ts, len(data), data[:10])


def holo_debug(api, stream_id=None, **kw):
    from .. import holoframe
    stream_id = stream_id or '+'.join(api.streams())
    for sid, ts, data in api.data(stream_id, **kw):
        print(sid, ts)
        try:
            t0 = time.time()
            data = holoframe.load(data)
            dt = time.time() - t0
            print(f'took {dt:.3g}s')
            for name, x in data.items():
                print(name, type(x).__name__, _pretty_val(x))
        except ValueError as e:
            if 'frame type' not in str(e):
                import traceback
                traceback.print_exc()
            print(sid, e)
        print()

def _pretty_val(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        detail = f"\n{x}" if x.size < 20 else f'(min={x.min():.3g}, max={x.max():.3g})'
        return f'{x.shape} {detail}'
    return str(x)[:50]


@util.async2sync
async def audio(api, stream_id, **kw):
    # kw2 = dict(last_entry_id=0)
    from .audio import AudioPlayer, unpack_audio
    with AudioPlayer() as player:
        async with api.data_pull_connect(stream_id, **kw) as ws:
            while True:
                for sid, ts, data in await ws.recv_data():
                    y, pos, sr, channels = unpack_audio(data)
                    # print(sid, ts, y.shape, pos, sr, channels)
                    if y is None:
                        continue
                    log.debug('read %s: %s (%s) pos=%d shape=%s q=%d', sid, ts, util.ts2datetime(ts).strftime('%c.%f'), pos, y.shape, player.q.qsize())
                    # print('read', sid, ts, util.ts2datetime(ts).strftime('%c.%f'), pos, y.shape, player.q.qsize())
                    player.write(y, pos, sr, channels)
                    time.sleep(1e-5)

# last_entry_id=1651848501191-0




# 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
gscale2 = '@%#*+=-:. '
gscale3 = "##&@$%*+=:-.  "
chars = np.array(list(gscale3)[::-1])  # setting the default as dark mode
def _ascii_image(img, width=60, invert=False):
    # resize the image
    w, h = img.size
    img = img.resize((width, int((h/w) * width * 0.55)))
    # convert image to greyscale format
    img = np.asarray(img.convert('L'))
    # quantize uint8 to ascii
    img = chars[::-1 if invert else 1][(img // (256 / len(chars))).astype(int)]
    return '\n'.join(''.join(map(str, xi)) for xi in img) 

def ascii_test(api, path, width=60, invert=False):
    print(_ascii_image(Image.open(path), width, invert))