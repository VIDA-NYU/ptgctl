import ptgctl
from ptgctl import holoframe

import timeit


def ti(cmd, data, n=100):
    print(cmd)
    dt = timeit.timeit(cmd, globals=dict(globals(), data=data), number=n)
    print(f'{ptgctl.util.pretty_bytes(len(data))}: {dt / n:.4g}s {dt / n / (len(data) / 1024**2):.3g}/mb - {n}x')


def main(n=100):
    api = ptgctl.API()
    streams = api.streams.ls()
    for sid in streams:
        for sid, ts, data in api.data(sid, last_entry_id='0'):
            d = holoframe.load(data)
            print(sid, ts, set(d))
            ti('holoframe.load(data)', data, n)
            ti('holoframe.load(data, metadata=True)', data, n)
            ti('holoframe.load(data, only_header=True)', data, n)



if __name__ == '__main__':
    import fire
    fire.Fire(main)