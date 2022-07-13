
'''Record data from the API to file.


'''
from __future__ import annotations
import os
import glob
import json
import tqdm
import zipfile
import cv2
import numpy as np

import ptgctl
from ptgctl import holoframe
from ptgctl import util
ptgctl.log.setLevel('WARNING')


def tqprint(*a, **kw):
    tqdm.tqdm.write(' '.join(map(str, a)), **kw)

IN_PATH =  '.'
OUT_PATH = 'output'

def _unzip(fname):
    with open(fname, 'rb') as f:
        with zipfile.ZipFile(f, 'r', zipfile.ZIP_STORED, False) as zf:
            for ts in tqdm.tqdm(sorted(zf.namelist()), desc=fname):
                with zf.open(ts, 'r') as f:
                    data = f.read()
                    yield ts, data


def _peek(it, n=1):
    '''Look at the first n elements of a csv.'''
    it = iter(it)
    first = [x for i, x in zip(range(n), it)]
    return first, (x for xs in (first, it) for x in xs)

def _iter_zip_data(rec_id, sid, in_path=IN_PATH):
    print(os.path.join(in_path, rec_id, sid, '*'))
    yield from (
        (ts, holoframe.load(data))
        for f in tqdm.tqdm(sorted(glob.glob(
            os.path.join(in_path, rec_id, sid, '*')
        )), desc=f'{rec_id}/{sid}')
        for ts, data in _unzip(f)
    )

# https://github.com/gradio-app/gradio/issues/1508#issuecomment-1154545730
def convert_video(rec_id, sid, key='image', fps=24, in_path=IN_PATH, out_path=OUT_PATH, start_time=None, overwrite=False):
    vid_fname = os.path.join(out_path, rec_id, f'{sid}.mp4')
    os.makedirs(os.path.dirname(vid_fname), exist_ok=True)
    if not overwrite and os.path.isfile(vid_fname):
        return vid_fname
    
    data_iter = (
        (util.parse_epoch_time(t), _process_frame(d[key])) for t, d in 
        _iter_zip_data(rec_id, sid, in_path=in_path)
    )
    first, data_iter = _peek(data_iter, n=1)
    size = next((x.shape[:2] for t, x in first), (640,480))
    start_time = util.parse_epoch_time(start_time) if start_time else next((t for t, x in first), 0)
    data_iter = ((t - start_time, x) for t, x in data_iter)

    #cc = 'avc1' #'mp4v' # 'I420' # "MJPG"
    #writer = cv2.VideoWriter(
    #    vid_fname, cv2.VideoWriter_fourcc(*cc), 
    #    fps, size[::-1], True)
    #try:
    #    t = 0
    #    ts, prev_im = next(data_iter)
    #    for ts, im in data_iter:
    #        while t < ts:
    #            writer.write(prev_im)
    #            t += 1.0 / fps
    #        prev_im = im
    #    writer.write(prev_im)
    #except StopIteration:
    #    print("Empty video")
    #finally:
    #    writer.release()
    try:
        ts, prev_im = next(data_iter)
        with video_writer(vid_fname, *size[::-1], fps) as writer:
            t = 0
            for ts, im in data_iter:
                while t < ts:
                    writer.write(prev_im.tobytes())
                    t += 1.0 / fps
                prev_im = im
            writer.write(prev_im.tobytes())
    except StopIteration:
        print("Empty video")
    return vid_fname


import contextlib
# src: https://stackoverflow.com/questions/61260182/how-to-output-x265-compressed-video-with-cv2-videowriter
@contextlib.contextmanager
def video_writer(fname, width, height, fps, vcodec='libx264'):
    import subprocess, shlex
    try:
        process = subprocess.Popen(shlex.split(
            f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} '
            f'-i pipe: -vcodec {vcodec} -pix_fmt yuv420p -crf 24 {fname}'
        ), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        yield process.stdin
    finally:
        pass


def _process_frame(im):
    if np.issubdtype(im.dtype, np.integer) and not np.issubdtype(im.dtype, np.uint8):
        im = (im / np.iinfo(im.dtype).max * 255).astype(np.uint8)
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im


def convert_json(rec_id, sid, in_path=IN_PATH, out_path=OUT_PATH, overwrite=False):
    fname = os.path.join(out_path, rec_id, f'{sid}.json')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if not overwrite and os.path.isfile(fname):
        return fname

    all_data = []
    for ts, d in _iter_zip_data(rec_id, sid, in_path=in_path):
        all_data.append({
            'timestamp': ts,
            **d,
        })

    with open(fname, 'w') as f:
        json.dump(all_data, f)
    return fname


def convert_audio(rec_id, sid, sr=None, in_path=IN_PATH, out_path=OUT_PATH, overwrite=False):
    fname = os.path.join(out_path, rec_id, f'{sid}.wav')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if not overwrite and os.path.isfile(fname):
        return fname

    data_iter = _iter_zip_data(rec_id, sid, in_path=in_path)
    first, data_iter = _peek(data_iter, n=10)
    sr = sr or next((d['sr'] for ts, d in first), 44100)
    channels = next((x.shape[1] if x.ndim > 1 else 1 for x in (d['audio'] for ts, d in first)), 1)

    first = 0
    lastpos = None

    import soundfile
    with soundfile.SoundFile(fname, 'w', samplerate=sr, channels=channels) as sf:
        for ts, d in data_iter:
            sr = sr or d['sr']
            pos = d['pos']
            y = d['audio']
            if lastpos:
                n_gap = min(max(0, pos - lastpos), sr * 2)
                if n_gap:
                    sf.write(np.zeros((n_gap, channels)))
            sf.write(y)
            lastpos = pos + len(y)
    return fname



def convert(rec_id, *sids, in_path=IN_PATH, **kw):
    rec_path = os.path.join(in_path, rec_id)
    in_path = os.path.dirname(rec_path)
    rec_id = os.path.basename(rec_path)
    # rec_id = os.path.basename(rec_path)
    sids = sids or [os.path.basename(d) for d in glob.glob(os.path.join(rec_path, '*'))]
    print('recording path:', rec_path, rec_id)
    print('sids:', sids)
    for sid in sids:
        print(sid)
        if sid in {'main', 'gll', 'glf', 'grf', 'grr', 'depthlt'}:
            convert_video(rec_id, sid, in_path=in_path, **kw)
        elif sid in {'hand', 'eye'}:
            convert_json(rec_id, sid, in_path=in_path, **kw)
        elif sid in {'mic0'}:
            convert_audio(rec_id, sid, in_path=in_path, **kw)



def convert_many(*fs, sids=None, path='.', **kw):
    fs = fs or glob.glob(os.path.join(path, '*'))
    for f in fs:
        print(f)
        convert(f, *sids or (), **kw)

if __name__ == '__main__':
    import fire
    fire.Fire(convert_many)
