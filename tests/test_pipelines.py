import numpy as np
from ptgctl import pipelines as P
import pytest



class Array(P.Block):
    def run(self, shape, fps=None):
        self.x = x = np.random.random(shape)
        for _ in self.read(fps=fps):
            yield x

class Multiply(P.Block):
    def run(self, n=2):
        for x, in self.read():
            yield x * n

class EveryN(P.Block):
    def run(self, n=2):
        for x, in self.read():
            if not self.processed_count % n:
                yield x

class AssertShape(P.Block):
    def run(self, shape):
        for x, in self.read():
            assert x.shape == shape
            yield x


class BlockDidRun(P.BlockContext):
    def on_exit(self, *x):
        b = self.block
        # assert b.processed_count > 0
        # assert b.generated_count > 0
        print(b, b.processed_count, b.generated_count)



@pytest.mark.parametrize("shape", [
    (),
    (1,),
    (10,),
    (10, 10),
    (10, 10, 10),
    (10, 10, 10, 10),
    (10, 10, 10, 10, 10),
])
@pytest.mark.parametrize("process", [False, True])
def test_basic(shape, process):
    import pickle
    contexts = [BlockDidRun()]
    pickle.dumps(contexts)
    with P.Graph() as g:
        x = Array(shape, contexts=contexts)
        xm = Multiply(is_process=process, contexts=contexts)(x)
        AssertShape(shape, contexts=contexts)(xm)
        EveryN(2)(xm)

    

    g.run(duration=1)
