'''Universal parser for Hololens messages.

This is here for convenience but won't be necessary for too much longer

'''
# from collections import namedtuple
import struct
from collections import defaultdict
import orjson
import numpy as np
from PIL import Image
import cv2




def load_all(read_data):#, dt_tol=0.1
    '''Take data read from the API and convert them into hololens frames.

    Arguments:
        read_data (list): Each item should be (stream_id, ts, data_bytes)
    '''
    time_steps = defaultdict(dict)
    for stream_id, ts, data_bytes in read_data:
        time_steps[stream_id].update(load(data_bytes))
        time_steps[stream_id]['timestamp'] = ts

    # time_steps = defaultdict(lambda: defaultdict(dict))
    # for stream_id, ts, data_bytes in read_data:
    #     time_steps[ts][stream_id].update(load(data_bytes))
    # # TODO merge close timesteps
    # time_steps = sorted(time_steps.items())
    # first = time_steps[0]

    return dict(time_steps)



def unpack(data, keys):
    return [nested_key(data, k) for k in keys]


def nested_key(data, key, default=...):
    d = data
    try:
        for ki in key.split('.'):
            d = d[int(ki) if isinstance(d, list) else ki]
        return d
    except KeyError:
        if default != ...:
            return default
        raise KeyError(f'{key!r} - {ki!r} not in {set(d) if isinstance(d, dict) else len(d) if isinstance(d, (list, tuple)) else d}')


def get_image(data):
    '''Given a loaded dict of hololens data, access an image instance.'''
    return next((
        data[k]['image']
        for k in ['main', 'glf', 'grf', 'glr', 'grr', 'depthlt']
        if k in data
    ), None)


class SensorType:
    PV = 0
    DepthLT = 1
    DepthAHAT = 2
    GLL = 3
    GLF = 4
    GRF = 5
    GRR = 6
    Accel = 7
    Gyro = 8
    Mag = 9
    numSensor = 10
    Calibration = 11
    # !!! these are circumstantial
    Mic = 172
    Hand = 34
    Eye = 34



header_dtype = np.dtype([
    ('version', np.uint8), 
    ('ftype', np.uint8),
])
header2_dtype = np.dtype([
    ('time', np.uint64),
    ('w', np.uint32),
    ('h', np.uint32),
    ('stride', np.uint32),
    ('info_size', np.uint32),
])
depth_dtype = np.dtype(np.uint16).newbyteorder('>')


def load(data, metadata=False, only_header=False):
    '''Parse any frame of data coming from the hololens.'''
    parse = ByteParser(data)
    read = not metadata  # disable reading images

    version, ftype = parse.pop(header_dtype)
    d = dict(frame_type=ftype)

    # special cases

    # hand+eye - no header
    if ftype in {34}: 
        if only_header:
            return d
        vals = parse.reset().pop_json()
        for k in ['left', 'right']:
            if k in vals and isinstance(vals[k], str):
                vals[k] = orjson.loads(vals[k])
        d.update(vals)
        return d
    # microphone
    if ftype == 172:
        d['sr'], channels, d['pos'] = parse.reset().unpack('<iiq')
        if only_header:
            return d
        d['audio'] = parse.pop(np.float32, (-1, channels), read=read)
        return d

    ts, w, h, stride, info_size = parse.pop(header2_dtype)
    im_size = w*h*stride
    d['time'] = ts

    if only_header:
        return d

    # image
    if ftype in {SensorType.PV, SensorType.GLF, SensorType.GRR, SensorType.GRF, SensorType.GLL}:
        yuv = ftype in {SensorType.PV}
        rot = -1 if ftype in {SensorType.GLF, SensorType.GRR} else 1 if ftype in {SensorType.GRF, SensorType.GLL} else 0
        d['image'] = parse.pop_image(w, h, stride, yuv=yuv, rot=rot, read=read)
    
        if info_size > 0:
            d['cam2world' if ftype in {SensorType.PV} else 'rig2world'] = parse.pop(np.float32, (4,4), T=True)
            if ftype in {SensorType.PV}:
                d['focalX'], d['focalY'] = parse.pop(np.float32, (2,))
                d['principalX'], d['principalY'] = parse.pop(np.float32, (2,))
        return d

    # depth
    if ftype in {SensorType.DepthLT, SensorType.DepthAHAT}:
        d['image'] = parse.pop(depth_dtype, (h, w), read=read)
        if info_size >= im_size:
            info_size -= im_size
            d['infrared'] = parse.pop(np.uint16, (h, w), read=read)
        if info_size > 0:
            d['rig2world'] = parse.pop(np.float32, (4,4), T=True)
        return d

    # sensors
    if ftype in {SensorType.Accel, SensorType.Gyro, SensorType.Mag}:
        d['data'] = parse.pop(np.float32, (h, w), im_size, read=read)
        timestamps = parse.pop(np.uint64, size=info_size)
        d['timestamps'] = (timestamps - timestamps[0]) // 100 + ts
        return d

    # calibration
    if ftype in {SensorType.Calibration}:
        d['lut'] = parse.pop(np.float32, (w * h, 3), read=read)
        d['rig2cam'] = parse.pop(np.float32, (4,4), T=True)
        return d

    raise ValueError(f"unknown frame type: {ftype}")


class ByteParser:
    def __init__(self, data):
        self.data = memoryview(data)
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    @property
    def total(self) -> int:
        return len(self.data)

    def _pop_bytes(self, size: int) -> memoryview:
        if size is None:  # if None, go to the end
            size = len(self.data) - self.offset
        i, j = self.offset, self.offset + size
        self.offset = j
        return self.data[i:j]

    def _read_np(self, x: memoryview, dtype, shape=None, T=False) -> np.ndarray:
        x = np.frombuffer(x, dtype)
        if shape:
            x = x.reshape(shape)
        elif x.size == 1:
            x = x.item()
        if T:
            x = x.T
        return x

    def _read_image(self, im: memoryview, w, h, rot=0, yuv=False) -> np.ndarray:
        im = np.array(Image.frombytes('L', (w, h), bytes(im)))
        if yuv:
            im = cv2.cvtColor(im[:,:-8], cv2.COLOR_YUV2RGB_NV12)
        if rot:
            im = np.rot90(im, rot)
        return im

    def _read_json(self, x: memoryview):
        return orjson.loads(str(x, 'ascii'))

    def pop(self, dtype, shape=None, size=None, read=True, **kw):
        x = self._pop_bytes(size or np_size(dtype, shape))
        if read:
            x = self._read_np(x, dtype, shape, **kw)
        return x

    def pop_image(self, w: int, h: int, stride: int, read=True, **kw):
        x = self._pop_bytes(w * h * stride)
        if read:
            x = self._read_image(x, w, h, **kw)
        return x

    def pop_json(self, size=None, read=True):
        x = self._pop_bytes(size)
        if read:
            x = self._read_json(x)
        return x

    def unpack(self, format) -> tuple:
        data = self._pop_bytes(struct.calcsize(format))
        return struct.unpack(format, data)

    def reset(self):
        self.offset = 0
        return self


def np_size(dtype, shape=None):
    '''Get the size of an array with data type and shape.'''
    mult = 1
    for s in shape or ():
        if s < 0:
            return None
        mult *= s
    return np.dtype(dtype).itemsize * mult
