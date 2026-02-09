import time
import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from tinygrad.engine.jit import TinyJit

# Define Sequential AND Sigmoid using tinygrad:
class Sequential:
    def __init__(self, *layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        # check that each sub-layer has parameters() defined and returns a list
        params = []
        for layer in self.layers:
            # If layer has a parameters() method, call it, otherwise assume no parameters.
            if hasattr(layer, 'parameters'):
                params += list(layer.parameters())
            if hasattr(layer, "weight"):
                params += [layer.weight]
            if hasattr(layer, "bias"):
                params += [layer.bias]
        return params

class Sigmoid:
    def __call__(self, x):
        return x.sigmoid()
    def parameters(self):
        return []

# Build model
nn = Sequential(
    Linear(512, 256),
    Sigmoid(),
    Linear(256, 128),
    Sigmoid(),
)

print(nn.parameters())

inp = Tensor.randn(2, 512)
opt = SGD(nn.parameters(), lr=0.01)

def f():
    opt.zero_grad()
    out = nn(inp)
    loss = out.sum()  # or other loss as needed
    loss.backward()
    opt.step()
    return

# Optionally JIT the function for maximum performance (compiles forward+backward+step)
@TinyJit
def f_jit():
    return f()

def benchmark(func, name=""):
    start = time.time()
    func()
    print(f"{name}: {time.time()-start:.4f}s")

def loop_many_jit(n):
    for _ in range(n):
        f_jit()

Tensor.training=True
benchmark(lambda: loop_many_jit(100_000), name="JIT Tinygrad Forward/Backward/Step x10k")