import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodiff import execute, ir_for, Feeder, Receiver
import autodiff
import os
from autodiff.nn import TransformerEncoder, SGD 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
import numpy as np

nn = TransformerEncoder(
    num_layers=6, 
    d_model=64,
    num_heads=4,
    ff_dim=128
)

idx = 0
def get_inp ():
    return np.full((2,64), 0.3, dtype=np.float32)

def save_params (*args):
    print(f"Number of params: {len(args)}")

opt = SGD(nn.parameters(), lr=0.01)
def f():
    opt.zero_grad()
    val = Feeder(lambda: get_inp(), shape=[2,64])
    res = nn(val)
    res.backward() 
    opt.step()

    #Receiver(lambda _: None, [res])

benchmark(lambda: ir_for(range(0, 500), f), name="Tracking nodes")
Receiver(save_params, opt.parameters, name="saving params")
benchmark(lambda: execute(), name="full exec")

