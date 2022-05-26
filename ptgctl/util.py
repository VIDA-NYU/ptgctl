import sys
import time
import json
import base64
import datetime
import asyncio
import functools
import inspect

import logging


NC='\033[0m'
def color(text, color='grey', brightness=0):
    return f'{color_code(color, brightness)}{text}{NC}'

def color_code(color='grey', brightness=0):
    i = 1 if brightness > 0 else 2 if brightness < 0 else 0
    c = COLOR_CODES[color.lower()]
    return f'\033[{i};{c}m'

COLOR_CODES = {
    'red':    30,
    'orange': 31,
    'green':  32,
    'yellow': 33,
    'blue':   34,
    'purple': 35,
    'cyan':   36,
    'grey':   37,
}

class ColorFormatter(logging.Formatter):
    LEVELS = {
        logging.DEBUG: color_code('purple', 0), 
        logging.INFO: color_code('grey', -1), 
        logging.WARNING: color_code('orange', 1), 
        logging.ERROR: color_code('red', 0), 
        logging.CRITICAL: color_code('red', 1), 
    }
    def format(self, record):
        return self.color(super().format(record), record.levelno)

    def color(self, text, level):
        color = self.LEVELS.get(level)
        return f'{color}{text}{NC}' if color else text

def getLogger(name=__name__.split('.')[0], level='info'):
    '''Get a logger.
    '''
    log = logging.getLogger(name)
    log.propagate = False
    if log.handlers:
        return log
    log_handler = logging.StreamHandler(sys.stderr)
    formatter = ColorFormatter('%(message)s')
    log_handler.setFormatter(formatter)
    log.addHandler(log_handler)
    log.setLevel(aslevel(level))
    def log_color(text, level):
        return formatter.color(text, aslevel(level))
    log.color = log_color
    return log

def aslevel(level):
    return logging._nameToLevel.get(
        level.upper() if isinstance(level, str) and level not in logging._nameToLevel else level, 
        level)



def async2sync(func):
    '''Wraps an async function with a synchronous call.'''
    @functools.wraps(func)
    def sync(*a, **kw):
        task = func(*a, **kw)
        return asyncio.run(task)
    sync.asyncio = func
    return sync


async def async_first_done(*unfinished):
    '''Returns when the first task finishes and cancels the rest. 
    
    This is used when both sending and receiving data and you interrupt one of them, they should all exit.
    '''
    try:
        finished, unfinished = await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
        return next((x for x in (t.result() for t in finished) if x is not None), None)
    finally:
        for task in unfinished:
            task.cancel()
        await asyncio.wait(unfinished)


def filternone(d):
    '''Filter None values from a dictionary. Useful for updating only a few fields.'''
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if v is not None}
    return d


def ts2datetime(rid):
    '''Convert a redis timestamp to a datetime object.'''
    return datetime.datetime.fromtimestamp(int(rid.split('-')[0])/1000)

def pack_entries(data):
    '''Pack multiple byte objects into a single bytearray with numeric offsets.'''
    entries = bytearray()
    offsets = []
    for d in [data] if not isinstance(data, (list, tuple)) else data:
        offsets.append(len(d))
        entries += d
    return offsets, entries

def unpack_entries(offsets, content):
    '''Unpack a single bytearray with numeric offsets into multiple byte objects.'''
    entries = []
    for (sid, ts, i), (_, _, j) in zip(offsets, offsets[1:] + [(None, None, None)]):
        entries.append((sid, ts, content[i:j]))
    return entries



class BoundModule:
    '''Acts as a method namespace for a class using a module as the namespace.
    
    The api object will be provided as the first argument to any function accessed from 
    this module.

    .. code-block:: python

        # greetings.py - a module of functions to bind to API

        def hello(api, your_name):
            print(api.name, 'hello', your_name)

        def hi(api):
            print(api.name, 'hi')
        
        # main.py

        class API:
            def __init__(self, name):
                self.name = name

            @property
            def greetings(self):
                import greetings
                return BoundModule(self, greetings)

        API().something.hello('friend')
        API().something.hi()

    '''
    _COPY_OVER = ['__doc__', '__name__', '__file__', '__package__']
    def __init__(self, parent, wrapped):
        self._self = parent
        self._wrapped = wrapped
        keys = getattr(wrapped, '__bind__', None) or getattr(wrapped, '__all__', None) or dir(wrapped)
        candidates = {k: getattr(wrapped, k, None) for k in keys}
        self._public = {
            name: f for name, f in candidates.items() 
            if self._is_public(f)
        }
        self._dir = list(self._public)
        # mimic - idk
        for k in self._COPY_OVER:
            setattr(self, k, getattr(self._wrapped, k))

    def _is_public(self, func):
        return is_public_func(func) and belongs_to_module(self._wrapped, func)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__name__}, {self._dir})'        

    def __dir__(self):
        return self._dir

    def __getattr__(self, name):
        if name not in self._public:
            raise AttributeError(name)
        return self._public[name].__get__(self._self)



def bound_module(get_module):
    '''Helper for bound modules. This will cache per-class for future use.
    
    .. code-block:: python

        class A:
            @bound_module
            def greetings(self):
                import greetings
                return greetings
    '''
    name = f'__{get_module.__name__}'
    @property
    @functools.wraps(get_module)
    def inner(self):
        # only create the bound module once
        try:
            return getattr(self, name)
        except AttributeError:
            bm = BoundModule(self, get_module(self))
            setattr(self, name, bm)
            return bm
    return inner


def is_public_func(func):
    name = getattr(func, '__name__', None)
    return callable(func) and name and not name.startswith('_')

def belongs_to_module(mod, func):
    return inspect.getmodule(func) == mod


# cli namespacing

class _NestedMetaClass(type):
    def __init__(self, name, bases, attrs):
        super().__init__(name, bases, attrs)
        self.__attr_name__ = '__{}'.format(name)

    # Works as a property that instantiates a nested class on first access.
    # the created instance will be used for subsequent access.
    def __get__(self, instance, owner=None):
        if instance is None:  # for class based access
            return self
        try:
            return getattr(instance, self.__attr_name__)
        except AttributeError:
            x = self(instance)
            setattr(instance, self.__attr_name__, x)
            return x

    # don't allow setting this property
    def __set__(self, instance, value):  # This is needed for Fire so that `ismethoddescriptor` is False
        raise TypeError()


class Nest(metaclass=_NestedMetaClass):
    '''This class lets you create method namespaces inside a class while still being able to access the original object.
    
    To access the parent object, use ``self.__``.

    .. code-block:: python

        class MyClass:
            x = 10
            class namespace(Nest):
                y = 15
                def asdf(self):
                    # self.__.x and self.x are the same value btw
                    return self.__.x + self.x + self.y  # 10 + 10 + 15

        import fire
        fire.Fire(MyClass)

    .. code-block:: bash

        python myclass.py nested asdf
    '''

    ROOT_ATTRS = True  # whether to check the parent object for attributes.
    def __init__(self, instance):
        # give it a name, like a function
        self.__name__ = self.__class__.__name__
        # this is the parent instance
        self.__ = instance
        # store the root instance
        root = instance
        if isinstance(instance, Nest):
            root = getattr(instance, '_root_', instance)
        self._root_ = root

    def __getattr__(self, key):
        if not self.ROOT_ATTRS:
            raise KeyError(key)
        return getattr(self.__, key)




class Token(dict):
    '''Wraps a JWT token and handles parsing the token information and checking its expiration.
    
    .. code-block:: python

        token = Token(token_str)

        if token:
            print(f'Your token is valid for {token.time_left.total_seconds():.0f} secs.')
            print(token)
            requests.get('/something', headers={'Authorization': f'Bearer {token}'})
        else:
            print('token is empty or has expired.')
    '''
    min_time_left = 10  # seconds
    def __init__(self, token=None):
        self.token = token = str(token or '')
        self.header, self.data, self.signature = jwt_decode(token) if token else ({}, {}, '')
        super().__init__(self.data)

        expires = self.get('exp')
        self.expires = datetime.datetime.fromtimestamp(expires) if expires else None
        self.min_time_left = datetime.timedelta(seconds=self.min_time_left)

    def __repr__(self):
        '''Show the token information, including expiration and data payload.'''
        if self.token is None:
            return 'Token(None)'
        return 'Token(time_left={}, {})'.format(self.time_left, super().__repr__())

    def __str__(self):
        '''Get the token as a string (can be passed in an Authorization header)'''
        return str(self.token or '')

    def __bool__(self):
        '''Check if the token exists and is not expired.'''
        return bool(self.token) and self.time_left > self.min_time_left

    @property
    def time_left(self) -> datetime.timedelta:
        '''Gets the amount of time left, as a datetime.timedelta object.'''
        return self.expires - datetime.datetime.now() if self.expires else datetime.timedelta(seconds=0)



def partdecode(x):
    '''Decode the token base64 string part as a dictionary. Split the token using ``'.'`` first.'''
    return json.loads(base64.b64decode(x + '===').decode('utf-8'))

# def partencode(x):
#     '''Encode the token dictionary payload as a base64 string.'''
#     return base64.b64encode(json.dumps(x).encode('utf-8')).decode('utf-8')

def jwt_decode(token):
    '''Decode a token into it's parts and parse the header, payload, and signature.'''
    header, data, signature = token.split('.')
    return partdecode(header), partdecode(data), signature

# def jwt_encode(header, data, signature=None):
#     '''Take a token's parts and convert it to a token. This is only really used for mocking a token for testing purposes.'''
#     header, data = partencode(header), partencode(data)
#     signature = signature or hash(str(header+data))  # default to dummy signature
#     return '.'.join((header, data, signature))

