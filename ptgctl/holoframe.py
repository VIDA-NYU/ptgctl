'''Universal parser for Hololens messages.

This is here for convenience but won't be necessary for too much longer

'''
# from collections import namedtuple
import struct
import cv2
from PIL import Image
import numpy as np
import orjson


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

    if ftype in {34}: # we interrupt this message to bring you hand+eye
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
        elif ftype in {SensorType.GLF, SensorType.GRR}:
            im = np.rot90(im, -1)
        elif ftype in {SensorType.GRF, SensorType.GLL}:
            im = np.rot90(im)
            
        rig2world = focal = None
        if info_size > 0:
            rig2world, data = np_pop(data, np.float32, (4,4))
            rig2world = rig2world.T
            if ftype in {SensorType.PV}:
                focal, data = np_pop(data, np.float32, (2,))

        return dict(frame_type=ftype, image=im, rig2world=rig2world, focal=focal)

    # depth

    if ftype in {SensorType.DepthLT, SensorType.DepthAHAT}:
        depth, data = np_pop(data, depth_dtype, (h, w), im_size)

        ab = cam2world = None
        if info_size >= im_size:
            info_size -= im_size
            ab, data = np_pop(data, np.uint16, (h, w), im_size)
        
        if info_size > 0:
            cam2world, data = np_pop(data, np.float32, (4,4))
            cam2world = cam2world.T
        return dict(frame_type=ftype, depth=depth, ab_image=ab, cam2world=cam2world)

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
        lut, data = np_pop(data, np.float32, (-1,3), im_size)
        rig2world, data = np_pop(data, np.float32, (4,4))
        rig2world = rig2world.T
        return dict(frame_type=ftype, lut=lut, rig2world=rig2world)

    raise ValueError(f"unknown frame type: {ftype}")


def np_read(data: bytes, dtype, *a, shape=None, **kw):
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
