import time
import queue
# from collections import deque
import struct
import numpy as np
import sounddevice as sd
from .. import util

log = util.getLogger(__name__, level='debug')


def pack_audio(y, pos, sr, channels):
    return struct.pack('<iiq', int(sr), int(channels), int(pos)) + y.tobytes()

def unpack_audio(data):
    if len(data) <= 16:
        return None, None, None, None
    # unpack audio
    header = data[:16]
    data = data[16:]
    sr, channels, pos = struct.unpack('<iiq', header)
    y = np.frombuffer(data, dtype=np.float32).reshape((-1, channels))
    return y, pos, sr, channels



# class Dequeue:
#     def __init__(self, maxsize):
#         self.queue = deque(maxlen=int(self.q_duration / self.block_duration))
#     def empty(self):
#         return not self.queue
#     def get(self, *a, **kw):
#         return self.get_nowait()
#     def get_nowait(self):
#         return self.queue.popleft() if self.queue else None
#     def put(self, x, *a, **kw):
#         self.queue.append(x)


class AudioBase:
    stream = None
    StreamCls = sd.Stream
    t0 = None
    offset = i = 0

    block_duration = 0.2
    q_duration = 10
    def __init__(self, **kw):
        self.kw = dict({'samplerate': 44100, 'channels': 1}, **kw)
        # only keep ~2 seconds buffer before fetching more
        qsize = int(self.q_duration / self.block_duration)
        self.q = queue.Queue(maxsize=qsize) 

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _init(self, **kw):
        s = self.stream
        kw = {k: v for k, v in kw.items() if v is not None}
        if s and all(self.kw.get(k) == v for k, v in kw.items()):
            return
        self.close()

        log.info(f'Updating stream params: {kw}')
        self.t0 = None
        self.i = 0
        self.sr = self.kw['samplerate']
        self.channels = self.kw['channels']
        log.info(f'Block size: {self.block_duration} {int(self.sr * self.block_duration)}')
        self.stream = self.StreamCls(
            **self.kw, dtype=np.float32, 
            blocksize=int(self.sr * self.block_duration),
            callback=self._callback_wrap)
        self.stream.start()

    def close(self):
        s = self.stream
        if s is not None:
            log.warning('closing stream')
            s.stop()
            log.debug('stopped')
            s.close()
            log.debug('closed')
        self.stream = None

    def _callback_wrap(self, buf, frames, t, status):
        if status:
            print('status:', status)
        return self._callback(buf, frames, t, status)

    def _callback(self, buf, frames, t, status):
        pass


class AudioRecorder(AudioBase):
    '''A class to record audio from a microphone to be uploaded to the API.
    
    .. code-block:: python

        from ptgctl.tools import audio

        with audio.AudioRecorder(device=device) as rec:
            async with api.data_push_connect(sid) as ws:
                while True:
                    y, pos = rec.read()
                    if y is None:
                        continue
                    await ws.send_data(audio.pack_audio(y, pos, rec.sr, rec.channels))
    
    '''
    StreamCls = sd.InputStream

    def read(self, sr=None, channels=None, block=True, timeout=None):
        '''Read an audio chunk from the microphone buffer.'''
        self._init(samplerate=sr, channels=channels)
        return self.q.get(block=block, timeout=timeout) or (None, None)

    im1=0
    def _callback(self, buf, frames, t, status):
        if self.t0 is None:
            self.t0 = t.inputBufferAdcTime
        if self.im1 is None:
            self.im1 = 0
        # i = round((t.inputBufferAdcTime - self.t0) * self.sr * self.channels)
        self.im1 += buf.size
        self.q.put((np.copy(buf), self.im1))
        s = self.q.qsize()
        if s > 1:
            print('qsize', s)
        # print(buf.shape, frames, self.im1)


class AudioPlayer(AudioBase):
    '''A class to play audio from the API.
    
    .. code-block:: python

        with AudioPlayer() as player:
            async with api.data_pull_connect(stream_id, **kw) as ws:
                while True:
                    for sid, ts, data in await ws.recv_data():
                        y, pos, sr, channels = unpack_audio(data)
                        if y is None:
                            continue
                        player.write(y, pos, sr, channels)

    '''
    StreamCls = sd.OutputStream
    offset = 0
    last_pos = 0

    def write(self, y, pos, sr=None, channels=None):
        '''Write an audio chunk to the output buffer.'''
        self._init(samplerate=sr, channels=channels)
        self.q.put((pos, y))
        #print(y.dtype, y.shape, y.min(), y.max())
        time.sleep(1e-3)

    def _callback(self, buf, frames, t, status):
        q = self.q
        buf.fill(0)
        time.sleep(1e-6)
        if q.empty():
            return
        if self.t0 is None:
            self.t0 = t.outputBufferDacTime

        # get the output buffer positions
        ch = self.channels
        ibuf = self.offset + round((t.outputBufferDacTime - self.t0) * self.sr * ch)
        jbuf = ibuf + frames * ch

        # allow fast-forward/rewind to skip blanks between recordings
        pos = q.queue[0][0]
        if jbuf < pos or pos < self.last_pos:  # pos is outside bounds
            # shift offset so that ibuf lines up with pos
            log.debug('\nFast-Forward / Rewind: %d to %d\n', self.last_pos, pos)
            self.offset += pos - ibuf
            ibuf = pos
            jbuf = ibuf + frames * ch
        self.last_pos = pos

        log.debug('output src: %d buffer [%d, %d]', pos, ibuf, jbuf)

        # Start copying audio samples to output buf
        while not q.empty() and jbuf > pos:
            # get next frame from queue
            pos, y = q.queue[0]
            end = pos + len(y) * ch
            # detect backwards skips
            if pos < self.last_pos:
                return
            self.last_pos = pos
            # get relative offsets between buffer and sample
            left, right = max(pos, ibuf), min(end, jbuf)
            if right > left:
                buf[(left - ibuf) // ch:(right - ibuf) // ch] = y[(left - pos) // ch:(right - pos) // ch]
            # if there's still some samples left over, don't pop them just yet
            if jbuf < end:
                break
            # if not, we can discard
            q.get_nowait()
