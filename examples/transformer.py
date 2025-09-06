from autodiff import execute, ir_for, Feeder, Receiver
import autodiff
from autodiff.nn import TransformerEncoder, SGD 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
import numpy as np

nn = TransformerEncoder(
    num_layers=1, 
    d_model=64,
    num_heads=4,
    ff_dim=128
)

idx = 0
def get_inp ():
    global idx
    if (idx+1) % 100 == 0:
        print(f"Get inp Idx: {idx+1}")
    idx += 1
    return np.full((2,64), 0.3, dtype=np.float32)

def save_params (*args):
    print(len(args))
    for arg in args:
        print(np.sum(arg))

opt = SGD(nn.parameters(), lr=0.01)
def f():
    opt.zero_grad()
    val = Feeder(lambda: get_inp(), shape=[2,64])
    res = nn(val)
    res.backward() 
    opt.step()

    Receiver(lambda: None, [res])

# In future release pass the idx
benchmark(lambda: ir_for(range(0, 100), f), name="Tracking nodes")

Receiver(save_params, opt.parameters, name="saving params")

benchmark(lambda: execute(), name="full exec")

