import time
import contextlib
import collections
import concurrent.futures as cf
import numpy as np



def first(xs):
    '''True if the first element is true'''
    return bool(next(iter(xs), True))


def _argstr(x):
    if isinstance(x, np.ndarray):
        return f'[shape={x.shape}]'
    return str(x)


class Block:
    '''This represents a task that has some inputs and outputs. It can run in either 
    a separate process or a separate thread. You connect the blocks together and they
    will pass their data to each other using queues.
    
    '''
    _worker = None
    _args, _kwargs = (), {}
    delay = 1/200.
    rate = None
    profile = False

    def __init__(self, *a, name=None, is_process=False, profile=None, contexts=None, **kw):
        self.name = name or self.__class__.__name__.lower()
        self._args, self._kwargs = a, kw
        self.in_queues = []
        self.in_states = []
        self.out_queues = []
        self.running = State().process_safe(is_process)
        self.error = State().process_safe(is_process)
        self.is_process = is_process

        # context managers/events
        profile = profile if profile is not None else self.profile
        contexts = [
            PrintFPS(10),
            Profile() if profile else None,
            IOCount(),
            *(contexts or ()),
        ]
        self._contexts = [c for c in contexts if c]

        if Graph.default:
            Graph.default.blocks.append(self)

    def __getstate__(self):
        return dict(self.__dict__, _worker=None)

    def __str__(self) -> str:
        return f"[Block({self.name}) {', '.join(map(_argstr, self._args + tuple(f'{k}={_argstr(v)}' for k, v in self._kwargs.items())))}]"

    def __call__(self, *inputs):
        for i in inputs:
            s = i.running.process_safe(i.is_process)
            q = mpDeque(maxlen=1) if self.is_process or i.is_process else tDeque(maxlen=1)
            q._name = self.name
            self.in_states.append(s)
            self.in_queues.append(q)
            i.out_queues.append(q)
        return self

    def run_worker(self, *a, **kw):
        try:
            print("Starting", self)
            with BlockContext.enter_all(self._contexts, self) as contexts:
                with contextlib.closing(self.run(*a, **kw)) as items:
                    for x in items:
                        for q in self.out_queues:
                            q.append(x)
                        for c in contexts:
                            c.on_write(x)

        except Exception as e:
            self.error(True)
            import traceback
            traceback.print_exc()
        # except BaseException: pass # ??
        finally:
            self.running(False)
            print("Ending", self)

    def read(self, when=first, fps=None, wait=True):
        in_states = self.in_states
        qs = self.in_queues
        throttle = Throttler(fps)
        while True:
            # check running state
            if not self.running:
                print(f"{self} was stopped.")
                break
            if not all(r for r in in_states):
                print(f"{self} stopped because its inputs stopped.", [bool(r) for r in in_states])
                break
            throttle()
            
            # check input queues
            if not when(q for q in qs):
                if not wait:
                    break
                time.sleep(self.delay)
                continue

            xs = [q.pop() if q else None for q in qs]
            for c in self._contexts:
                c.on_read(*xs)
            yield xs


    def run(self, *a, fps=None, **kw):
        try:
            self.init(*a, **kw)
            for xs in self.read(fps=fps):
                yield self.process(*xs)
        finally:
            self.finish()

    def init(self): pass

    def process(self, x, *xs):
        return x

    def finish(self): pass

    def start(self):
        self.running(True)
        self.error(False)
        if self.is_process:
            import multiprocessing as mp
            worker = mp.Process(target=self.run_worker, name=str(self), args=self._args, kwargs=self._kwargs, daemon=True)
        else:
            import threading
            worker = threading.Thread(target=self.run_worker, name=str(self), args=self._args, kwargs=self._kwargs, daemon=True)
        worker.start()
        self._worker = worker

    def close(self):
        self.running(False)

    def join(self):
        if self._worker is None:
            return
        self.running(False)
        if self._worker.is_alive():
            print('joining', self)
            self._worker.join(timeout=5)
        self._worker = None
        self.running(False)


class Graph:
    _delay = 1
    def __init__(self):
        self.blocks = []

    def __enter__(self):
        Graph.default, self.previous = self, Graph.default
        return self

    def __exit__(self, *a):
        if self.previous:
            Graph.default, self.previous = self.previous, None

    @contextlib.contextmanager
    def run_scope(self):
        '''Spawn when entering the context and join when exiting.'''
        blocks = self.blocks
        try:
            for b in blocks:
                b.start()
            yield self.still_running
        finally:
            for b in blocks:
                b.close()
            for b in blocks:
                b.close()  # just in case ???
                b.join()

    def still_running(self):
        blocks = self.blocks
        return any(b.running for b in blocks) and not any(b.error for b in blocks)

    def wait(self, duration=None):
        t0 = time.time()
        while self.still_running():
            time.sleep(self._delay)
            if duration and time.time() - t0 > duration:
                break

    def run(self, duration=None):
        '''Run multiple blocks'''
        with self.run_scope():
            self.wait(duration)

        

Graph.default = Graph()


class State:
    '''Wraps state management'''
    value = False
    def __init__(self):
        self.callbacks = []

    def __call__(self, value=True):
        if self.value == value: return
        self.value = value
        for f in self.callbacks:
            f(value)

    def __bool__(self):
        return bool(self.value)

    def process_safe(self, needs=True):
        if needs and not isinstance(self, ProcessSafe):
            import multiprocessing as mp
            self._value = mp.Value('i', self.value)
            self.__class__ = ProcessSafe
        return self

class ProcessSafe(State):
    @property
    def value(self):
        return self._value.value

    @value.setter
    def value(self, value):
        self._value.value = value



class tDeque(collections.deque):
    _name = None

class mpDeque(collections.deque):
    '''A quick hack to utilize deque across processes.'''
    _name = None
    def __init__(self, *a, **kw):
        import multiprocessing as mp
        self._q = mp.Queue()
        super().__init__(*a, **kw)
    def _pull(self):
        while not self._q.empty():
            # print(self._name, 'dropped')
            super().append(self._q.get())

    def pop(self, *a, **kw):
        self._pull()
        return super().pop(*a, **kw)

    def append(self, value):
        self._q.put(value)

    def __len__(self):
        self._pull()
        return super().__len__()

    def __bool__(self):
        self._pull()
        return super().__bool__()


class Throttler:
    def __init__(self, rate=None):
        self.rate = rate
        self.t_last = 0

    def __call__(self):
        rate = self.rate
        if not rate:
            return 
        tnew = time.time()
        dt = 1./rate - (tnew - self.t_last)
        time.sleep(max(dt, 1e-6))
        self.t_last = time.time()

class Scheduler:
    t0 = i = 0
    def __init__(self, func, interval=10):
        self.func = func
        self.interval = interval

    def reset(self):
        self.t0 = time.time()
        self.i = 0

    def __call__(self, *a, **kw):
        i = (time.time() - self.t0) // self.interval
        if i and i > self.i:
            self.func(*a, **kw)
            self.i = i



class BlockContext:
    def __call__(self, block):
        self.block = block
        self.on_bind(block)
        return self

    _context = None
    def __enter__(self):
        self._context = contextlib._GeneratorContextManager(self.context, [self.block], {})
        self._context.__enter__()
        self.on_enter()

    def __exit__(self, t, e, tb):
        if e:
            self.on_error(e)
        self.on_exit()
        if self._context:
            self._context.__exit__(t, e, tb)
        self._context = None

    def context(self, block): yield
    def on_bind(self, block): pass
    def on_enter(self): pass
    def on_read(self, *xs): pass
    def on_write(self, x): pass
    def on_error(self, e): pass
    def on_exit(self): pass

    @staticmethod
    @contextlib.contextmanager
    def enter_all(contexts, block):
        with contextlib.ExitStack() as stack:
            yielded = [stack.enter_context(c(block)) for c in contexts]
            yield contexts


class Profile(BlockContext):
    def context(self, block):
        import pyinstrument
        p = pyinstrument.Profiler()
        p.start()
        try:
            yield 
        finally:
            p.stop()
            print('\n' + '-'*20 + '\n' + str(block) + '\n' + '-'*20 + '\n' + p.output_text(color=True))


class PrintFPS(BlockContext):
    def __init__(self, interval=60):
        self.interval = interval

    def on_enter(self):
        self.t0 = self.ti = time.time()

    def on_write(self, *x):
        dt = time.time() - self.t0
        if time.time() - self.ti > self.interval:
            print(f'{self} - {self.i / dt} fps - {[len(q) for q in self.block.qs]} in qs')
            self.ti = time.time()


class IOCount(BlockContext):
    def on_bind(self, block):
        self.on_enter()

    def on_enter(self):
        self.block.processed_count = 0
        self.block.generated_count = 0

    def on_read(self, *x):
        self.block.processed_count += 1

    def on_write(self, *x):
        self.block.generated_count += 1


class LambdaContext(BlockContext):
    def __init__(self, **events):
        for name, event in events.items():
            if hasattr(self, name):
                setattr(self, name, event)
            else:
                raise AttributeError(name)


# some block implementations


class Multiplex(Block):  # XXX: NOT FUNCTIONALLY COMPLETE - self.__func is the problem...
    '''Let's you split up a computation over multiple processes.
    
    .. code-block::
        def compute(**kw):
            model = Model()
            def process_image(x):
                return model(x)
            return process_image
        Multiplex(compute)
    '''
    def run(self, func, max_workers=16, **kw):
        futures = collections.deque()
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(
                max_workers=max_workers, 
                initializer=self.__initializer,
                initargs=(func, kw)) as pool:
            for xs in self.read(when=lambda xs: all(xs) or futures and futures[0].done()):
                if not all(x is None for x in xs):
                    futures.append(pool.submit(self.__func, *xs))
                while futures and futures[0].done():
                    yield futures.popleft().result()

    def __initializer(self, func, kw):
        self.__func = func(**kw)


# some test blocks

class Inc(Block):
    def run(self, imax=10, interval=0.2):
        i = 0
        for _ in self.read():
            if i > imax:
                break
            yield i
            i += 1
            time.sleep(interval)

class Mult(Block):
    def process(self, i):
        return i * 2

class Debug(Block):
    def run(self, func=None):
        for x, in self.read():
            print(func(x) if func else x)
            yield x


class Constant(Block):
    def run(self, x, fps=None):
        for () in self.read(fps=fps):
            yield x


def main():
    with Graph() as g:
        inc = Inc()
        m1 = Mult()(inc)
        m2 = Mult(is_process=True)(m1)
        dbg = Debug()(m2)
    g.run()

if __name__ == '__main__':
    import fire
    fire.Fire(main)