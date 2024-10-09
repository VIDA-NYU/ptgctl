'''This contains functions to display values from the API in interesting ways.

'''
import os
import contextlib
import io
import json as json_
import time
import tqdm
import asyncio
import datetime
import numpy as np
from .. import util

log = util.getLogger(__name__, 'debug')


@util.async2sync
@util.interruptable
async def imshow(api, stream_id, delay=1, api_output='jpg', stream_format=None, **kw):
    '''Show a video stream from the API.'''
    import cv2
    from PIL import Image
    from .. import holoframe
    async with api.data_pull_connect(stream_id, output=api_output, input=stream_format, time_sync_id=0, ack=True, **kw) as ws:
        t0 = time.time()
        i = 0
        last_epoch = time.time()
        while True:
            entries = await ws.recv_data()
            t_now = time.time()
            inst_fps = len(entries) / (t_now - last_epoch)

            for sid, ts, data in entries:
                i += 1
                if api_output:
                    im = np.array(Image.open(io.BytesIO(data)))
                else:
                    im = holoframe.load(data)['image']
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                ts_frame = util.ts2datetime(ts)
                latency = datetime.datetime.now() - ts_frame
                text = f"{ts_frame.strftime('%c.%f')} [fps={inst_fps:.1f} avg fps={i / (time.time() - t0):.1f}, {latency} latency]"
                # print(sid, len(data), ts, text)
                # cv2.putText(im, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
                cv2.putText(im, text, (10, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                cv2.imshow(sid, im)

            last_epoch = t_now
            if cv2.waitKey(delay + (not len(entries)) * 50) & 0xFF == ord('q'):
                break


def imshow1(api, stream_id, **kw):#, raw_holo=False
    '''Show a single frame of a stream.'''
    import cv2
    from PIL import Image
    # if raw_holo:
    #     from .. import holoframe
    for sid, ts, data in api.data(stream_id, output='jpg', **kw):
        # if raw_holo:
        #     im = holoframe.load(data)['image']
        # else:
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



@util.async2sync
@util.interruptable
async def json(api, stream_id, **kw):
    async with api.data_pull_connect(stream_id, **kw) as ws:
        while True:
            for sid, ts, data in await ws.recv_data():
                print(f'{sid}: {ts}')
                try:
                    print(json_.loads(data.decode('utf-8')))
                except json_.decoder.JSONDecodeError:
                    import traceback
                    traceback.print_exc()
                    print("could not decode:", data)


@util.async2sync
@util.interruptable
async def raw(api, stream_id, utf=False, **kw):
    async with api.data_pull_connect(stream_id, **kw) as ws:
        while True:
            for sid, ts, data in await ws.recv_data():
                print(f'{sid}: {ts}')
                try:
                    print(data.decode('utf-8') if utf else data)
                except json_.decoder.JSONDecodeError:
                    import traceback
                    traceback.print_exc()
                    print("could not decode:", data)


@util.async2sync
@util.interruptable
async def file(api, stream_id, out_dir='', include_timestamps=False, **kw):
    os.makedirs(out_dir or '.', exist_ok=True)
    async with api.data_pull_connect(stream_id, **kw) as ws:
        with contextlib.ExitStack() as stack:
            files = {}
            pbars = {}
            while True:
                for sid, ts, data in await ws.recv_data():
                    if sid not in files:
                        files[sid] = stack.enter_context(open(os.path.join(out_dir, f'{sid}.txt'), 'w'))
                        pbars[sid] = tqdm.tqdm(desc=sid)
                    files[sid].write(f"{f'{ts}:' if include_timestamps else ''}{data.decode('utf-8')}\n")
                    pbars[sid].update()
                    pbars[sid].set_description(f'{sid}: {str(data)[:20]}')


@util.async2sync
@util.interruptable
async def fps(api, stream_id, **kw):
    import collections
    sids = stream_id.split('+')
    async with api.data_pull_connect(stream_id, **kw) as ws:
        last = collections.defaultdict(lambda: 0.0)
        while True:
            for sid, ts, _ in await ws.recv_data():
                t = util.parse_epoch_time(ts)
                dt = t - last[sid]
                print(f'{sid}: {ts}', f'∆{dt:.2g}s fps={1/(dt):.2g}' if last[sid] else '')
                last[sid] = t
            if len(sids)>1:
                print(f'rel to {sids[0]}', ' '.join([f'{s}: {last[s]  - last[sids[0]]:.3f}s' for s in sids[1:]]))


@util.async2sync
@util.interruptable
async def update(api, stream_id, **kw):
    from ptgctl import holoframe
    import tqdm
    async with api.data_pull_connect(stream_id, **kw) as ws:
        pbar = tqdm.tqdm()
        while True:
            for sid, ts, data in await ws.recv_data():
                pbar.set_description(f'{sid}: {util.parse_time(ts).strftime("%c")}')
                pbar.update()



def test(api, stream_id=None, **kw):
    if not stream_id:
        import json
        return json.dumps(api.streams(), indent=4)
    for sid, ts, data in api.data(stream_id, **kw):
        print(sid, ts, len(data), data[:10])


def holo_debug(api, stream_id=None, **kw):
    from .. import holoframe
    stream_id = stream_id or '+'.join(api.streams.ls())
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
async def debug_holo_stream(api, stream_id, **kw):
    '''Show a video stream from the API.'''
    from .. import holoframe
    async with api.data_pull_connect(stream_id, **kw) as ws:
        while True:
            for sid, ts, data_bytes in await ws.recv_data():
                data = holoframe.load(data_bytes)
                print(sid, ts, len(data_bytes), {k: getattr(x, 'shape', None) or x for k, x in data.items()})


@util.async2sync
async def audio(api, stream_id, **kw):
    # kw2 = dict(last_entry_id=0)
    from .audio import AudioPlayer, unpack_audio
    with AudioPlayer() as player:
        async with api.data_pull_connect(stream_id, latest=False, **kw) as ws:
            while True:
                for sid, ts, data in await ws.recv_data():
                    y, pos, sr, channels = unpack_audio(data)
                    # print(sid, ts, y.shape, pos, sr, channels)
                    if y is None:
                        continue
                    log.debug('read %s: %s (%s) pos=%d shape=%s q=%d', sid, ts, util.ts2datetime(ts).strftime('%c.%f'), pos, y.shape, player.q.qsize())
                    # print('read', sid, ts, util.ts2datetime(ts).strftime('%c.%f'), pos, y.shape, player.q.qsize())
                    player.write(y, pos, sr, channels)
                    await asyncio.sleep(1e-5)

# last_entry_id=1651848501191-0



import cv2
import orjson
import collections
import supervision as sv

@util.async2sync
@util.interruptable
async def objects(api, stream_id='main', track_stream_id='detic:image', obj_stream_id='detic:image:misc', **kw):
    from .. import holoframe
    box_ann = sv.BoxAnnotator(text_scale=0.8, text_padding=1)
    mask_ann = sv.MaskAnnotator()
    imsize = None
    frames = collections.deque(maxlen=64)
    det_frame = None
    track_frame = None
    async with api.data_pull_connect([stream_id, track_stream_id, obj_stream_id], latest=False, **kw) as ws:
        while True:
            for sid, ts, data in await ws.recv_data():
                ts = util.parse_epoch_time(ts)
                if sid == 'main':
                    d = holoframe.load(data)
                    frame = d['image'][:,:,::-1]
                    frames.append([ts, frame])
                    imsize = np.array([frame.shape[1], frame.shape[0]])
                if imsize is None:
                    print("No image shape")
                    continue

                if sid == track_stream_id:
                    track_frame_ = draw_object_json(data, ts, imsize, frames, mask_ann, box_ann)
                    if track_frame_ is not None:
                        track_frame = track_frame_
                if sid == obj_stream_id:
                    det_frame_ = draw_object_json(data, ts, imsize, frames, mask_ann, box_ann)
                    if det_frame_ is not None:
                        det_frame = det_frame_
                
                if sid in {obj_stream_id, track_stream_id}:
                    cv2.imshow('objects', np.concatenate([x for x in [track_frame, det_frame] if x is not None]))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

def draw_object_json(data, ts, imsize, frames, mask_ann, box_ann):
    data = orjson.loads(data)

    # get the frame
    track_frame = next((f for t, f in frames if abs(t - ts) < 1/50), None)
    if track_frame is None:
        track_frame = min(frames, key=lambda x: abs(x[0]-ts))

    if not data or track_frame is None:
        return track_frame
    # [ {"label": } ]

    # get the detections
    labels = np.array([d['label'] for d in data])
    confidence = np.array([d['confidence'] for d in data])
    
    xyxy = np.array([d['xyxyn'] for d in data])
    xyxy[:, 0] *= imsize[0]
    xyxy[:, 1] *= imsize[1]
    xyxy[:, 2] *= imsize[0]
    xyxy[:, 3] *= imsize[1]
    mask = np.array([contour2mask(d['segment'], imsize) for d in data]) if 'segment' in data[0] else None
    track_id = np.array([d['segment_track_id'] for d in data]) if 'segment_track_id' in data[0] else None
    detections = sv.Detections(xyxy=xyxy, mask=mask, class_id=track_id, tracker_id=track_id, confidence=confidence)

    states = []
    state_confs = []
    for d in data:
        dstate = d.get('state')
        if dstate:
            max_state, max_conf = max(dstate.items(), key=lambda x: x[1])
            states.append(max_state)
            state_confs.append(max_conf)
        else:
            states.append(None)
            state_confs.append(0)

    text = [
        f'{labels[i]}({confidence[i]:.0%})'+(
        f' - {states[i]}({state_confs[i]:.0%})' if states[i] else '')
        for i in range(len(labels))
    ]

    # draw the frame
    track_frame = track_frame.copy()
    track_frame = mask_ann.annotate(track_frame, detections)
    track_frame = box_ann.annotate(track_frame, detections, text)
    return track_frame


# Implement the contour2mask function
def contour2mask(normalized_contour, im_size):
    mask = np.zeros((im_size[1], im_size[0]), dtype=np.uint8)  # Initialize an empty mask
    for segment in normalized_contour:
        segment = (segment * im_size).astype(np.int32)
        cv2.fillPoly(mask, [segment], 1)  # Fill the mask using the contour
    return mask


# @util.async2sync
# async def debug_holo_stream(api, stream_id, **kw):
#     '''Show a video stream from the API.'''
#     from contextlib import ExitStack
#     import tqdm
#     from .. import holoframe
#     async with api.data_pull_connect(stream_id, **kw) as ws:
#         pbars = {}
#         with ExitStack() as stack:
#             while True:
#                 for sid, ts, data_bytes in await ws.recv_data():
#                     if sid not in pbars:


#                     data = holoframe.load(data_bytes)
#                     print(sid, ts, len(data_bytes), {k: getattr(x, 'shape', None) or x for k, x in data.items()})


# @util.async2sync
# async def stream(api, stream_id, **kw):
#     '''Show a video stream from the API.'''
#     from .. import holoframe
#     async with api.data_pull_connect(stream_id, **kw) as ws:
        


# 70 levels of gray
# gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
# gscale2 = '@%#*+=-:. '
gscale3 = "##&@$%*+=:-.  "
chars = np.array(list(gscale3)[::-1])  # setting the default as dark mode
def ascii_image(img, width=60, height=None, invert=False, preserve_aspect=True):
    if img is None:
        return ''
    if isinstance(img, np.ndarray):
        from PIL import Image
        img = Image.fromarray(img)
    # resize the image
    w, h = img.size
    img = img.resize(_aspect(width, height, w, h, preserve_aspect))
    # convert image to greyscale format
    img = np.asarray(img.convert('L'))
    # quantize uint8 to ascii
    img = chars[::-1 if invert else 1][(img // (256 / len(chars))).astype(int)]
    return '\n'.join(''.join(map(str, xi)) for xi in img) 


def _aspect(w, h, w_im, h_im, preserve=True):
    aspect = w_im / h_im
    w_aspect = h * aspect if h else w
    h_aspect = w * aspect if w else h
    w = w or w_aspect
    h = h or h_aspect
    if preserve:
        w = min(w, w_aspect)
        h = min(h, h_aspect)
    return int(w), int(h)

def ascii_test(api, path, width=60, invert=False):
    from PIL import Image
    print(ascii_image(Image.open(path), width, invert))
