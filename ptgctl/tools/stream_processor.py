'''

 - have a continuous live running process
 - run one-off jobs


RawRecorder.run_continuously
RawRecorder.run
RawRecorder.run_while_active


'''
from __future__ import annotations
# import heartrate; heartrate.trace(browser=True)
import functools
from typing import AsyncIterator, cast
import os
import time
import asyncio
import orjson
import contextlib
import datetime
import tqdm

import numpy as np
from PIL import Image

import ptgctl
from ptgctl import holoframe
from ptgctl.util import parse_epoch_time, async2sync



class Context:
    def __init__(self, **kw):
        self.kw = kw

    def context(self):
        raise NotImplementedError
        yield
    async def acontext(self):
        raise NotImplementedError
        yield

    __context = __acontext = None
    def __enter__(self, *a, **kw):
        self.__context = contextlib.contextmanager(self.context)(*a, **kw, **self.kw)
        return cast(self.__class__, self.__context.__enter__())

    def __exit__(self, *a):
        if self.__context:
            return self.__context.__exit__(*a)
        self.__context=None

    def __aenter__(self, *a, **kw):
        self.__acontext = contextlib.asynccontextmanager(self.acontext)(*a, **kw, **self.kw)
        return cast(self.__class__, self.__acontext.__aenter__())

    def __aexit__(self, *a):
        if self.__acontext:
            return self.__acontext.__aexit__(*a)
        self.__acontext=None

    def write(self, id, data):
        raise NotImplementedError


class StreamReader(Context):
    def __init__(self, api, streams, recording_id=None, raw=False, progress=True, merged=False, **kw):
        super().__init__(streams=streams, **kw)
        self.api = api
        self.recording_id = recording_id
        self.raw = raw
        self.merged = merged
        self.progress = progress

    async def acontext(self, streams, fullspeed=None, last=None, timeout=5000) -> 'AsyncIterator[StreamReader]':
        self.replayer = None
        rid = self.recording_id
        if rid:
            async with self.api.recordings.replay_connect(
                    rid, '+'.join(streams), 
                    fullspeed=fullspeed, 
                    prefix=f'{rid}:'
            ) as self.replayer:
                async with self.api.data_pull_connect('+'.join(f'{rid}:{s}' for s in streams), last=last, timeout=timeout) as self.ws:
                    yield self
            return

        async with self.api.data_pull_connect('+'.join(streams)) as self.ws:
            yield self


    async def watch_replay(self):
        if self.replayer is not None:
            try:
                await self.replayer.done()
            finally:
                self.running = False

    async def __aiter__(self):
        self.running = True
        import tqdm
        pbar = tqdm.tqdm()
        while self.running:
            data = await self.ws.recv_data()
            if self.recording_id:
                data = [(sid[len(self.recording_id)+1:], t, x) for (sid, t, x) in data]
            if self.merged:
                yield holoframe.load_all(data)
                pbar.update()
            else:
                for sid, t, x in data:
                    yield (sid, t, x) if self.raw else (sid, parse_epoch_time(t), holoframe.load(x))
                    pbar.update()


class StreamWriter(Context):
    def __init__(self, api, streams, test=False, **kw):
        super().__init__(streams=streams, **kw)
        self.api = api
        self.test = test

    async def acontext(self, streams):
        if self.test:
            yield self
            return
        async with self.api.data_push_connect('+'.join(streams), batch=True) as self.ws:
            yield self

    async def write(self, data):
        if self.test:
            print(data)
            return
        await self.ws.send_data(data)


import cv2
class ImageOutput:#'avc1', 'mp4v', 
    def __init__(self, src, fps, cc='mp4v', show=None):
        self.src = src
        self.cc = cc
        self.fps = fps
        self._show = not src if show is None else show
        self.active = self.src or self._show

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self._w:
            self._w.release()
        self._w = None
        if self._show:
            cv2.destroyAllWindows()
    async def __aenter__(self): return self.__enter__()
    async def __aexit__(self, *a): return self.__exit__(*a)

    def output(self, im):
        if self.src:
            self.write_video(im)
        if self._show:
            self.show_video(im)

    _w = None
    def write_video(self, im):
        if not self._w:
            self._w = cv2.VideoWriter(
                self.src, cv2.VideoWriter_fourcc(*self.cc),
                self.fps, im.shape[:2][::-1], True)
            if not self._w.isOpened():
                raise RuntimeError(f"Video writer did not open - probably because {self.cc}")
        self._w.write(im)

    def show_video(self, im):
        cv2.imshow('output', im)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration


def maybe_profile(func, min_time=20):
    @functools.wraps(func)
    def inner(*a, profile=False, **kw):
        if not profile:
            return func(*a, **kw)
        from pyinstrument import Profiler
        p = Profiler()
        t0 = time.time()
        try:
            with p:
                return func(*a, **kw)
        finally:
            if time.time() - t0 > min_time:
                p.print()
    return inner



class Processor:
    def __init__(self, api=None):
        name = self.__class__.__name__
        self.api = api or ptgctl.API(username=name, password=name)

    def call_async(self, streams):
        raise NotImplementedError

    @classmethod
    @maybe_profile
    @async2sync
    async def run(cls, *a, continuous=False, **kw):
        self = cls()
        while True:
            try:
                await self.call_async(*a, **kw)
                if not continuous:
                    return
            except KeyboardInterrupt:
                print('byee!')
                return
            except Exception as e:
                if not continuous:
                    raise
                print(f'{type(e).__name__}: {e}')



class Yolo3D(Processor):
    output_prefix = 'yolo3d'
    image_box_keys = ['xyxy', 'confidence', 'class_id']
    world_box_keys = ['xyz_tl', 'xyz_br', 'xyz_tr', 'xyz_bl', 'xyzc', 'confidence', 'class_id']
    min_dist_secs = 1

    def __init__(self):
        super().__init__()
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        self.labels = np.asarray([
            self.model.names.get(i, i) for i in range(max(self.model.names))
        ])

    async def call_async(self, prefix=None, replay=None, fullspeed=None, **kw):
        from ptgctl.pt3d import Points3D
        self.data = {}

        prefix = prefix or ''
        out_prefix = f'{prefix}{self.output_prefix}'
        in_sids = ['main', 'depthlt', 'depthltCal']
        out_sids = [f'{out_prefix}:image', f'{out_prefix}:world']

        async with StreamReader(self.api, in_sids, recording_id=replay, fullspeed=fullspeed, merged=True) as reader, \
                   StreamWriter(self.api, out_sids, test=True) as writer, \
                   ImageOutput(None, None, show=True) as imout:
            async def _stream():
                data = holoframe.load_all(self.api.data(f'{replay if replay else ""}:depthltCal'))
                if replay:
                    data = {k[len(replay)+1:]:v for k, v in data.items()}
                self.data.update(data)

                async for data in reader:
                    self.data.update(data)
                    # if 'depthltCal' not in self.data:
                    #     self.data.update(holoframe.load_all(self.api.data('depthltCal')) or {})

                    try:
                        main, depthlt, depthltCal = [self.data[k] for k in in_sids]
                        rgb = main['image']

                        # check time difference
                        mts = main['timestamp']
                        dts = depthlt['timestamp']
                        secs = parse_epoch_time(mts) - parse_epoch_time(dts)
                        if abs(secs) > self.min_dist_secs:
                            raise KeyError(f"timestamps too far apart main={mts} depth={dts} ∆{secs:.3g}s")
                        
                        # create point transformer
                        pts3d = Points3D(
                            rgb, depthlt['image'], depthltCal['lut'],
                            depthlt['rig2world'], depthltCal['rig2cam'], main['cam2world'],
                            [main['focalX'], main['focalY']], [main['principalX'], main['principalY']])
                    except KeyError as e:
                        tqdm.tqdm.write(f'KeyError: {e} {set(self.data)}')
                        continue

                    xyxy = self.model(rgb).xyxy[0].numpy()
                    confs = xyxy[:, 4]
                    class_ids = xyxy[:, 5].astype(int)
                    xyxy = xyxy[:, :4]
                    xyz_tl, xyz_br, xyz_tr, xyz_bl, xyzc, dist = pts3d.transform_box(xyxy[:, :4])
                    valid = dist < 5  # make sure the points aren't too far

                    # print(xyxy)
                    # print(class_ids, confs)

                    # await writer.write([
                    #     self.dump(self.image_box_keys, [xyxy, confs, class_ids], valid),
                    #     self.dump(self.world_box_keys, [xyz_tl, xyz_br, xyz_tr, xyz_bl, xyzc, confs, class_ids], valid),
                    # ])
                    imout.output(draw_boxes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), xyxy, [
                        f'{self.labels[l]} {c:.0%} [{x:.0f},{y:.0f},{z:.0f}]' 
                        for l, c, (x,y,z) in zip(class_ids, confs, xyzc)
                    ]))

            await asyncio.gather(_stream(), reader.watch_replay())

    def dump(self, keys, xs, valid):
        return orjson.dumps([
            dict(zip(keys, xs), label=self.labels[int(xs[-1])])
            for xs in zip(*[x[valid] for x in xs])
        ], option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)


def draw_boxes(im, boxes, labels):
    for xy, label in zip(boxes, labels):
        xy = list(map(int, xy))
        im = cv2.rectangle(im, xy[:2], xy[2:4], (0,255,0), 2)
        im = cv2.putText(im, label, xy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return im


def draw_text_list(img, texts, i=-1, tl=(10, 50), scale=0.5, space=50, color=(255, 255, 255), thickness=1):
    for i, txt in enumerate(texts, i+1):
        cv2.putText(
            img, txt, 
            (int(tl[0]), int(tl[1]+scale*space*i)), 
            cv2.FONT_HERSHEY_COMPLEX, 
            scale, color, thickness)
    return img, i


class ActionClip(Processor):
    output_prefix = 'action_clip'
    prompts = {
        'tools': 'a photo of a {}',
        'ingredients': 'a photo of a {}',
        'instructions': '{}',
    }

    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        import torch, clip
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.tokenize = clip.tokenize

    async def _wait_for_redis_value(self, initial_id=None, delay=1):
        '''Wait for an active or changed recipe ID from redis.'''
        while True:
            self.current_id = rec_id = await self._get_id()
            if rec_id != initial_id:
                return rec_id
            await asyncio.sleep(delay)

    async def run_while_active(self, recipe_id, , prefix=None, replay=None, fullspeed=None):
        assert recipe_id, "You must provide a recipe ID, otherwise we have nothing to compare"
        # load the recipe from the api
        recipe = self.api.recipes.get(recipe_id)
        texts = {k: recipe[k] for k, _ in self.prompts.items()}
        z_texts = {k: self.encode_text(recipe[k], prompt) for k, prompt in self.prompts.items()}

        out_keys = set(texts)
        out_sids = [f'{prefix or ""}{self.output_prefix}:{k}' for k in out_keys]
        async with StreamReader(self.api, [f'{prefix or ""}main'], recording_id=replay, fullspeed=fullspeed) as reader, \
                   StreamWriter(self.api, out_sids, test=True) as writer, \
                   ImageOutput(None, None, show=True) as imout:
            async for _, xs in reader:
                if recipe_id and self.current_id != recipe_id:
                    break
                for t, d in xs:
                    # encode the image and compare to text queries
                    z_image = self.encode_image(d['image'])
                    writer.write([
                        self._bundle(texts[k], self.compare_image_text(z_image, z_texts[k])[0]) 
                        for k in out_keys
                    ])
                    imout.output()

    def encode_text(self, texts, prompt_format=None):
        '''Encode text prompts. Returns formatted prompts and encoded CLIP text embeddings.'''
        toks = self.tokenize([prompt_format.format(x) for x in texts] if prompt_format else texts).to(self.device)
        z = self.model.encode_text(toks)
        z /= z.norm(dim=-1, keepdim=True)
        return z, texts

    def encode_image(self, image):
        '''Encode image to CLIP embedding.'''
        image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        z_image = self.model.encode_image(image)
        z_image /= z_image.norm(dim=-1, keepdim=True)
        return z_image

    def compare_image_text(self, z_image, z_text):
        '''Compare image and text similarity (not sure why the 100, it's from the CLIP repo).'''
        return (100.0 * (z_image @ z_text.T)).softmax(dim=-1)

    def _bundle(self, text, similarity):
        '''Prepare text and similarity to be uploaded.'''
        return dict(zip(text, similarity.tolist()))



class BaseRecorder(Processor):
    class Writer(Context):
        def __init__(self, name, store_dir, **kw):
            super().__init__(**kw)
        def context(self, sample, t_start): yield self
        def write(self, t, data): raise NotImplementedError

    raw = False
    STORE_DIR = 'recordings'
    STREAMS: list|None = None

    def new_recording_id(self):
        return datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

    recording_id = None

    async def call_async(self, streams=None, recording_id=None, replay=None, fullspeed=None, progress=True, store_dir=None, **kw):
        store_dir = os.path.join(store_dir or self.STORE_DIR, recording_id or self.new_recording_id())
        os.makedirs(store_dir, exist_ok=True)

        if not streams:
            streams = self.api.streams.ls()
            if self.STREAMS:
                streams = [s for s in self.api.streams.ls() if any(s.endswith(k) for k in self.STREAMS)]
        elif isinstance(streams, str):
            streams = streams.split('+')

        writers = {}
        with contextlib.ExitStack() as stack:
            async with StreamReader(self.api, recording_id=replay, streams=streams, progress=progress, fullspeed=fullspeed, raw=self.raw) as reader:
                async def _stream():
                    async for sid, t, x in reader:
                        if recording_id and self.recording_id != recording_id:
                            break

                        if sid not in writers:
                            writers[sid] = stack.enter_context(
                                self.Writer(sid, store_dir, sample=x, t_start=t, **kw))

                        writers[sid].write(t, x)

                await asyncio.gather(_stream(), reader.watch_replay())

class RawRecorder(BaseRecorder):
    raw=True
    class Writer(Context):
        def __init__(self, name, store_dir, **kw):
            super().__init__(**kw)
            self.fname = os.path.join(store_dir, f'{name}.zip')

        def context(self, sample, t_start):
            import zipfile
            print("Opening zip file:", self.fname)
            with zipfile.ZipFile(self.fname, 'w', zipfile.ZIP_STORED, False) as self.writer:
                yield self

        def write(self, id, data):
            self.writer.writestr(id, data)


class VideoRecorder(BaseRecorder):
    class Writer(Context):
        def __init__(self, name, store_dir, sample, t_start, fps=15, vcodec='libx264', crf='23',  **kw):
            super().__init__(**kw)
            fname = os.path.join(store_dir, f'{name}.mp4')
            
            self.prev_im = sample['image'][:,:,::-1].tobytes()
            self.t_start = t_start
            h, w = sample['image'].shape[:2]

            self.fps = fps
            self.cmd = (
                f'ffmpeg -y -s {w}x{h} -pixel_format bgr24 -f rawvideo -r {fps} '
                f'-i pipe: -vcodec {vcodec} -pix_fmt yuv420p -crf {crf} {fname}')

        def context(self):
            import subprocess, shlex, sys
            process = subprocess.Popen(
                shlex.split(self.cmd), 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=sys.stderr)
            self.writer = process.stdin

            self.t = 0
            try:
                print("Opening video ffmpeg process:", self.cmd)
                yield self
            except BrokenPipeError as e:
                print(f"Broken pipe writing video: {e}")
                if process.stderr:
                    print(process.stderr.read())
                raise e
            finally:
                print('finishing')
                if process.stdin:
                    process.stdin.close()
                process.wait()
                print('finished')

        def write(self, ts, data):
            im = data['image'][:,:,::-1].tobytes()
            while self.t < ts - self.t_start:
                self.writer.write(self.prev_im)
                self.t += 1.0 / self.fps
            self.writer.write(im)
            self.t += 1.0 / self.fps
            self.prev_im = im


class AudioRecorder(BaseRecorder):
    class Writer(Context):
        def __init__(self, name, store_dir, **kw):
            self.fname = os.path.join(store_dir, f'{name}.wav')
            super().__init__(**kw)

        def context(self, sample, **kw):
            x = sample['audio']
            self.channels = x.shape[1] if x.ndim > 1 else 1
            self.lastpos = None
            import soundfile
            print("Opening audio file:", self.fname)
            with soundfile.SoundFile(self.fname, 'w', samplerate=sample['sr'], channels=self.channels) as self.sf:
                yield self

        def write(self, t, d):
            pos = d['pos']
            y = d['audio']
            if self.lastpos:
                n_gap = min(max(0, pos - self.lastpos), d['sr'] * 2)
                if n_gap:
                    self.sf.write(np.zeros((n_gap, self.channels)))
            self.lastpos = pos + len(y)


class JsonRecorder(BaseRecorder):
    raw=True
    class Writer(Context):
        def __init__(self, name, store_dir, **kw):
            super().__init__(**kw)
            self.fname = os.path.join(store_dir, f'{name}.wav')
            
        def context(self, **kw):
            self.i = 0
            print("Opening json file:", self.fname)
            with open(self.fname, 'wb') as self.fh:
                self.fh.write(b'[\n')
                try:
                    yield self
                finally:
                    self.fh.write(b'\n]\n')

        def write(self, ts, d):
            if self.i:
                self.fh.write(b',\n')
            self.fh.write(orjson.dumps(
                dict(holoframe.load(d), timestamp=ts), 
                option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY))
            self.i += 1


# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.generic):
#             return obj.item()
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super().default(obj)  


VIDEO_STREAMS = ['main', 'gll', 'glf', 'grf',  'grr']
AUDIO_STREAMS = ['mic0']
JSON_STREAMS = ['imuaccel', 'imugyro', 'imumag', 'hand', 'eye', 'yolo3d:v1']


def run(api, name, continuous=False):
    streams = api.streams.ls()
    RawRecorder.run.send(name, streams, continuous=continuous)
    VideoRecorder.run.send(name, [s for s in streams if s in VIDEO_STREAMS], continuous=continuous)
    AudioRecorder.run.send(name, [s for s in streams if s in AUDIO_STREAMS], continuous=continuous)
    JsonRecorder.run.send(name, [s for s in streams if s in JSON_STREAMS], continuous=continuous)

    Yolo3D.run.send(prefix=None)
    ActionClip.run.send()


def run_background(api):
    return run(api, None, continuous=False)



if __name__ == '__main__':
    import fire
    fire.Fire({
        'video': VideoRecorder,
        'audio': AudioRecorder,
        'json': JsonRecorder,
        'raw': RawRecorder,
        'yolo': Yolo3D,
        'clip': ActionClip,
    })
