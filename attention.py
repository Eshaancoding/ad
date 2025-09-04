from autodiff import execute, ir_for, Feeder, Receiver, pg
import autodiff
from autodiff.graph.tensor import Tensor
from autodiff.nn import SGD 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
import numpy as np

from pprint import pprint

"""
nn = MultiHeadAttention(
    d_model=512,
    num_heads=4
)
"""

nn = TransformerEncoder(2, 512, 4, 1024)

idx = 0
def get_inp ():
    global idx
    if (idx+1) % 100 == 0:
        print(f"Get inp Idx: {idx+1}")
    idx += 1
    return np.full((2,512), 1.0, dtype=np.float32)

def print_inp (res):
    pass

def save_params (param_one, param_two, param_three, param_four):
    print(param_one.shape)
    print(param_two.shape)
    print(param_three.shape)
    print(param_four.shape)

opt = SGD(nn.parameters(), lr=0.01)

pprint(nn.parameters())
print()

inp = Tensor.randn([2,512])
def f():
    opt.zero_grad()
    res = nn(inp)

    res.backward() 
    opt.step()

    Receiver(print_inp, [res], name="Printing res")

# In future release pass the idx
benchmark(lambda: ir_for(range(0, 100), f), name="Tracking nodes")

pprint(opt.parameters)

#Receiver(save_params, opt.parameters, name="saving params")

benchmark(lambda: execute(), name="full exec")
