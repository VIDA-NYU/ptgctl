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




def load_all(read_data, dt_tol=0.1):
    '''Take data read from the API and convert them into hololens frames.

    Arguments:
        read_data (list): Each item should be (stream_id, ts, data_bytes)
    '''
    time_steps = defaultdict(dict)
    for stream_id, ts, data_bytes in read_data:
        time_steps[stream_id].update(load(data_bytes))

    # time_steps = defaultdict(lambda: defaultdict(dict))
    # for stream_id, ts, data_bytes in read_data:
    #     time_steps[ts][stream_id].update(load(data_bytes))
    # # TODO merge close timesteps
    # time_steps = sorted(time_steps.items())
    # first = time_steps[0]

    return dict(time_steps)



def unpack(data, keys):
    return [nested_key(data, k) for k in keys]


def nested_key(data, key):
    d = data
    try:
        for ki in key.split('.'):
            ki = int(ki) if isinstance(d, list) else ki
            # print(ki, set(d))
            d = d[ki]
            # print(type(d))
        return d
    except KeyError:
        raise KeyError(f'{key!r} - {ki!r} not in {set(d) if isinstance(d, dict) else len(d) if isinstance(d, (list, tuple)) else d}')


def get_image(data):
    '''Given a loaded dict of hololens data, access an image instance.'''
    for k in ['main', 'glf', 'grf', 'glr', 'grr']:
        im = data.get(k)
        if im is not None:
            return im


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


SensorStreamMap = {
    SensorType.PV: 'camera',
    SensorType.DepthLT: 'depthlt',
    SensorType.DepthAHAT: 'depthahat',
    SensorType.GLL: 'camlr', 
    SensorType.GLF: 'camlf',
    SensorType.GRF: 'camrf',
    SensorType.GRR: 'camrr',
    SensorType.Accel: 'accel',
    SensorType.Gyro: 'gyro',
    SensorType.Mag: 'mag',
    SensorType.numSensor: 'numsensor',
    SensorType.Calibration: 'calibration',
}



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


def load(data):
    '''Parse any frame of data coming from the hololens.'''
    full_data = data
    data = memoryview(data)

    (version, ftype), data = np_pop(data, header_dtype)

    # special cases

    if ftype in {34}: # hand+eye - no header
        return dict(orjson.loads(full_data.decode('ascii')), frame_type=ftype)
    if ftype == 172:
        data = full_data
        meta, data = split(data, 16)
        sr, channels, pos = struct.unpack('<iiq', meta)
        samples = np.frombuffer(data, dtype=np.float32).reshape((-1, channels))
        return dict(audio=samples, sr=sr, pos=pos)

    (ts, w, h, stride, info_size), data = np_pop(data, header2_dtype)
    im_size = w*h*stride

    # image

    if ftype in {SensorType.PV, SensorType.GLF, SensorType.GRR, SensorType.GRF, SensorType.GLL}:
        im, data = split(data, im_size)
        im = np.array(Image.frombytes('L', (w, h), bytes(im)))

        if ftype in {SensorType.PV}:
            im = cv2.cvtColor(im[:,:-8], cv2.COLOR_YUV2RGB_NV12)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif ftype in {SensorType.GLF, SensorType.GRR}:
            im = np.rot90(im, -1)
        elif ftype in {SensorType.GRF, SensorType.GLL}:
            im = np.rot90(im)
        d = dict(frame_type=ftype, image=im)
            
        rig2world = focal = None
        if info_size > 0:
            x2world, data = np_pop(data, np.float32, (4,4))
            x2world = x2world.T
            d.update({'cam2world' if ftype in {SensorType.PV} else 'rig2world': x2world})
            if ftype in {SensorType.PV}:
                focal, data = np_pop(data, np.float32, (2,))
                principal_pt, data = np_pop(data, np.float32, (2,))
                d.update(dict(focal=focal, principal_point=principal_pt))

        return d

    # depth

    if ftype in {SensorType.DepthLT, SensorType.DepthAHAT}:
        depth, data = np_pop(data, depth_dtype, (h, w), im_size)
        d = dict(frame_type=ftype, depth=depth)

        ab = rig2world = None
        if info_size >= im_size:
            info_size -= im_size
            ab, data = np_pop(data, np.uint16, (h, w), im_size)
            d.update(dict(infrared=ab))
        
        if info_size > 0:
            rig2world, data = np_pop(data, np.float32, (4,4))
            rig2world = rig2world.T
            d.update(dict(rig2world=rig2world))
        
        return d

    # sensors

    if ftype in {SensorType.Accel, SensorType.Gyro, SensorType.Mag}:
        sensorData, data = np_pop(data, np.float32, (h, w), im_size)
        timestamps, data = np_pop(data, np.uint64, size=info_size)
        timestamps = (timestamps - timestamps[0]) // 100 + ts
        return dict(
            frame_type=ftype,
            **({SensorStreamMap[ftype]: sensorData}),
            timestamps=timestamps)

    # calibration

    if ftype in {SensorType.Calibration}:
        # assert stride == 4  # just checking
        lut, data = np_pop(data, np.float32, (w * h, 3))
        rig2cam, data = np_pop(data, np.float32, (4,4))
        rig2cam = rig2cam.T
        return dict(frame_type=ftype, lut=lut, rig2cam=rig2cam)

    raise ValueError(f"unknown frame type: {ftype}")




def np_read(data: bytes, dtype, shape=None):
    '''Reads a numpy array of type and shape from the start of a byte array.'''
    x = np.frombuffer(data, dtype)
    return x.reshape(shape) if shape else x.item() if x.size == 1 else x

def split(data, l):
    '''split an array at an index.'''
    return data[:l], data[l:]

def np_size(dtype, shape=None):
    '''Get the size of an array with data type and shape.'''
    mult = 1
    for s in shape or ():
        if s < 0:
            raise ValueError("Can't get absolute size for a flexible shape array.")
        mult *= s
    return np.dtype(dtype).itemsize * mult

def np_pop(data, dtype, shape=None, size=None):
    '''Read a numpy array from a byte array and chop them from the start of the array.'''
    x, leftover = split(data, size or np_size(dtype, shape))
    return np_read(x, dtype, shape), leftover
