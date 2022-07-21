'''
python -m ptgctl.tools.local_record_stream video coffee-test-1 main \
  | ffmpeg -y -s 720x480 -pixel_format bgr24 -f rawvideo -r 30 \
    -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 coffee-test-1.main.mp4

'''
import os
import glob
import tqdm
import zipfile

import numpy as np
import cv2

import ptgctl
from ptgctl import holoframe
from ptgctl import util
ptgctl.log.setLevel('WARNING')


def _peek(it, n=1):
    '''Look at the first n elements of a csv.'''
    it = iter(it)
    first = [x for i, x in zip(range(n), it)]
    return first, (x for xs in (first, it) for x in xs)

def iter_zip_data(rec_id, sid, in_path='.', load=True):
    root = os.path.abspath(os.path.join(in_path, rec_id, sid))
    if not os.path.isdir(root):
        raise OSError(f"{root} does not exist.")
    yield from (
        (ts, holoframe.load(data) if load else data)
        for f in sorted(glob.glob(
            os.path.join(in_path, rec_id, sid, '*')
        ))
        for ts, data in _unzip(f)
    )

def _unzip(fname):
    with open(fname, 'rb') as f:
        with zipfile.ZipFile(f, 'r', zipfile.ZIP_STORED, False) as zf:
            for ts in sorted(zf.namelist()):
                with zf.open(ts, 'r') as f:
                    data = f.read()
                    yield ts, data

def _resample(it, fps=None):
    # dont resample
    if not fps:
        yield from it
        return
    # resample with constant fps
    try:
        prev_ts, prev_im = next(it)
        t = 0
        for ts, im in it:
            while t < ts:
                #print(prev_ts, ts-t)
                yield prev_ts, prev_im
                t += 1.0 / fps
            
            prev_ts = ts
            prev_im = im
        yield prev_ts, prev_im
    except StopIteration:
        print("Empty stream")

def _process_video_frame(im, scale=None, norm=False):
    if scale:
        im = (im * scale).astype(im.dtype)
    if norm:
        im = im / im.max()

    # convert int32 to uint8
    if not np.issubdtype(im.dtype, np.uint8):
        if np.issubdtype(im.dtype, np.integer):
            im = im.astype(float) / np.iinfo(im.dtype).max
        im = (im * 255).astype(np.uint8)

    # grayscale to gray rgb
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    # convert rgb -> bgr
    else:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im



def iter_stream(rec_id, sid, key=None, in_path='.', fps=None, start_time=None, relative_time=True, process=(lambda x:x), **kw):
    key = sid2key[sid] if key is True else key
    data_iter = (
        (util.parse_epoch_time(t), process(d[key] if key else d, **kw)) for t, d in 
        iter_zip_data(rec_id, sid, in_path=in_path)
    )

    # optionally make the timestamps relative to the start time
    if start_time:
        if isinstance(start_time, str):
            start_time = util.parse_epoch_time(start_time)
    elif relative_time:
        first, data_iter = _peek(data_iter, n=1)
        start_time = next((t for t, x in first), 0)
    if start_time:
        data_iter = ((t - start_time, x) for t, x in data_iter)

    yield from _resample(data_iter, fps)

def iter_video_stream(rec_id, sid, key='image', get_size=False, **kw):
    it = iter_stream(rec_id, sid, key=key, process=_process_video_frame, **kw)
    
    if get_size:
        first, it = _peek(it, n=1)
        h, w = next((x.shape[:2] for _, x in first), (760, 428))
        print(f'{h}x{w}')
        return

    yield from it
    


sid2key = {
    **{k: 'image' for k in ('main', 'depthlt', 'depthahat', 'gll', 'glf', 'grf', 'grr')},
    **{k: 'data' for k in ('imuaccel', 'imumag', 'imugyro')},
    **{k: 'audio' for k in ('mic0')},
}


def output(kind, rec_id, sid, progress=False, out_dir=None, **kw):
    it = (
        istream_video(rec_id, sid, **kw) 
        if kind == 'video' else
        istream(rec_id, sid, **kw))
    it = tqdm.tqdm(it) if progress else it

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for t, d in enumerate(it):
            1/0
    else:
        # write bytes to stdout
        for t, d in it:
            os.write(1, d.tobytes())


istream = iter_stream
istream_video = iter_video_stream

if __name__ == '__main__':
    import fire
    fire.Fire(output)
