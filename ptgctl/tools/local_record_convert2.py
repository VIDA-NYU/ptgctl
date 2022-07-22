
'''Record data from the API to file.


'''
from __future__ import annotations
import warnings
import os
import sys
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
from . import record_output
_peek = record_output._peek


def tqprint(*a, **kw):
    tqdm.tqdm.write(' '.join(map(str, a)), **kw)

IN_PATH =  '.'
OUT_PATH = 'output'


# https://github.com/gradio-app/gradio/issues/1508#issuecomment-1154545730
def convert_video(rec_id, sid, key='image', fps=24, in_path=IN_PATH, out_path=OUT_PATH, overwrite=False, **kw):
    vid_fname = os.path.join(out_path, rec_id, f'{sid}.mp4')
    os.makedirs(os.path.dirname(vid_fname), exist_ok=True)
    if not overwrite and os.path.isfile(vid_fname):
        return vid_fname

    data_iter = record_output.iter_video_stream(
        rec_id, sid, key, fps=fps, in_path=in_path, **kw)
    first, data_iter = _peek(data_iter, n=1)
    size = next((x.shape[:2] for t, x in first), (760, 4280))

    with video_writer(vid_fname, *size[::-1], fps) as writer:
        for ts, im in data_iter:
            writer.write(im.tobytes())
    return vid_fname


import contextlib
# src: https://stackoverflow.com/questions/61260182/how-to-output-x265-compressed-video-with-cv2-videowriter
@contextlib.contextmanager
def video_writer(fname, width, height, fps, vcodec='libx264', crf='23'):
    import subprocess, shlex
    cmd = (
        f'ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} '
        f'-i pipe: -vcodec {vcodec} -pix_fmt yuv420p -crf {crf} {fname}'
    )
    print(cmd)
    process = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
    try:
        yield process.stdin
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

def convert_json(rec_id, sid, in_path=IN_PATH, out_path=OUT_PATH, overwrite=False):
    fname = os.path.join(out_path, rec_id, f'{sid}.json')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if not overwrite and os.path.isfile(fname):
        return fname

    all_data = []
    for ts, d in record_output.iter_zip_data(rec_id, sid, in_path=in_path):
        all_data.append({
            'timestamp': ts,
            **{
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in d.items()
            },
        })

    with open(fname, 'w') as f:
        json.dump(all_data, f)
    return fname


def convert_imu_json(rec_id, sid, in_path=IN_PATH, out_path=OUT_PATH, overwrite=False):
    fname = os.path.join(out_path, rec_id, f'{sid}.json')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if not overwrite and os.path.isfile(fname):
        print('skipping')
        return fname

    all_ts = []
    all_data = []
    for ts, d in record_output.iter_zip_data(rec_id, sid, in_path=in_path):
        timestamps = d['timestamps']
        data = d['data']
        if len(timestamps) != len(data):
            warnings.warn(f"timestamps and {sid} data not equal length {len(timestamps)} != {len(data)}")
        
        all_ts.extend(timestamps[:len(data)].tolist())
        all_data.extend(data[:len(timestamps)].tolist())

    with open(fname, 'w') as f:
        json.dump(all_data, f)
    return fname


def convert_audio(rec_id, sid, sr=None, in_path=IN_PATH, out_path=OUT_PATH, overwrite=False):
    fname = os.path.join(out_path, rec_id, f'{sid}.wav')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if not overwrite and os.path.isfile(fname):
        return fname

    data_iter = record_output.iter_zip_data(rec_id, sid, in_path=in_path)
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
        try:
            print(sid)
            if sid in {'main', 'gll', 'glf', 'grf', 'grr', 'depthlt'}:
                f=convert_video(rec_id, sid, in_path=in_path, scale=40 if sid == 'depthlt' else None, **kw)
            elif sid in {'hand', 'eye'} or sid in {'gllCal', 'glfCal', 'grfCal', 'grrCal', 'depthltCal'}:
                f=convert_json(rec_id, sid, in_path=in_path, **kw)
            elif sid in {'imuaccel', 'imugyro', 'imumag'}:
                f=convert_imu_json(rec_id, sid, in_path=in_path, **kw)
            elif sid in {'mic0'}:
                f=convert_audio(rec_id, sid, in_path=in_path, **kw)
            else:
                print("skipping", sid)
                continue
            print(f)
        except Exception:
            import traceback
            traceback.print_exc()



def convert_many(*fs, sids=None, path='.', **kw):
    fs = fs or glob.glob(os.path.join(path, '*'))
    for f in fs:
        print(f)
        convert(f, *sids or (), **kw)

if __name__ == '__main__':
    import fire
    fire.Fire(convert_many)
