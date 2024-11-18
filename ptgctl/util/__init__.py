import datetime
import asyncio
import functools

from .log import *
from .cli import *
from .token import *



def aslist(x):
    return x if isinstance(x, (list, tuple)) else [x] if x is not None else []

# data parsing

def pack_entries(data: list, sid=None, ts=None) -> tuple:
    '''Pack multiple byte objects into a single bytearray with numeric offsets.'''
    entries = bytearray()
    offsets = []
    offset = 0
    for d in aslist(data):
        offset += len(d)
        offsets.append(offset)
        entries += d
    if sid:
        sid = aslist(sid)
        ts = ts and aslist(ts)
        assert len(sid) == len(offsets) and (not ts or len(ts) == len(offsets)), (len(offsets), len(sid), len(ts))
        offsets = list(zip(sid, ts, offsets)) if ts else list(zip(sid, offsets))
    return offsets, entries

def unpack_entries(offsets: list, content: bytes) -> list:
    '''Unpack a single bytearray with numeric offsets into multiple byte objects.'''
    entries = []
    for (sid, ts, i), (_, _, j) in zip(offsets, offsets[1:] + [(None, None, None)]):
        entries.append((sid, ts, content[i:j]))
    return entries



def parse_time(tid: str):
    '''Convert a redis timestamp to a datetime object.'''
    return datetime.datetime.fromtimestamp(parse_epoch_time(tid))

def parse_epoch_time(tid: str):
    '''Convert a redis timestamp to epoch seconds.'''
    return int(tid.split('-')[0])/1000

ts2datetime = parse_time  # deprecated

def format_time(dt: datetime.datetime):
    return format_epoch_time(dt.timestamp())

def format_epoch_time(tid: float, tlast=None):
    tms = int(tid * 1000)
    i = 0
    if tlast and tms == int(tlast * 1000):
        i = int((tid*1000 - tms)*100)
    return f'{tms}-{i}'


# misc


def filternone(d: dict):
    '''Filter None values from a dictionary. Useful for updating only a few fields.'''
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if v is not None}
    return d



def interruptable(func):
    @functools.wraps(func)
    def wrap(*a, **kw):
        try:
            return func(*a, **kw)
        except KeyboardInterrupt:
            print('\nInterrupted.')
    return wrap


# asyncio


def async2sync(func):
    '''Wraps an async function with a synchronous call.'''
    @functools.wraps(func)
    def sync(*a, **kw):
        return asyncio.run(func(*a, **kw))
    sync.asyncio = func
    return sync

def async_run_safe(future):
    loop = asyncio.get_event_loop()

    # import signal
    # def ask_exit(signame, loop):
    #     print("got signal %s: exit" % signame)
    #     loop.stop()
    # for signame in {'SIGINT', 'SIGTERM'}:
    #     loop.add_signal_handler(getattr(signal, signame), functools.partial(ask_exit, signame, loop))
    task = asyncio.ensure_future(future)
    try:
        return loop.run_until_complete(task)
    except KeyboardInterrupt:
        print('Interrupted asyncio loop')
        task.cancel()
        loop.run_forever()
        task.exception()
        raise
    finally:
        loop.close()

async def async_first_done(*unfinished):
    '''Returns when the first task finishes and cancels the rest. 
    
    This is used when both sending and receiving data and you interrupt one of them, they should all exit.
    '''
    finished, unfinished = await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
    try:
        return next((x for x in (t.result() for t in finished) if x is not None), None)
    finally:
        for task in unfinished:
            task.cancel()
        await asyncio.wait(unfinished)



def pretty_bytes(b, scale=1000, names=['b', 'kb', 'mb', 'gb', 'tb']):
    return next((
            f'{b / (scale**i):.1f}{n}' 
            for i, n in enumerate(names) 
            if b / (scale**(i+1)) < 1
        ), 
        f'{b / (scale**(len(names)-1))}{names[-1]}')
