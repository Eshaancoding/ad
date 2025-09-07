import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodiff import execute, ir_for, Feeder, Receiver, print_graph
import autodiff
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD 
from autodiff.nn.activations.relu import ReLU
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
from autodiff.print_graph import pg
import numpy as np

autodiff.graph.tensor.is_testing = True

nn = Sequential(
    Linear(255, 255),
    Sigmoid(),
    Linear(255, 255)
)

def print_inp (res):
    pass

def save_params (*args):
    pass
    #print(len(args))
    #for arg in args:
        #print(arg)

opt = SGD(nn.parameters(), lr=0.01)
def f():
    opt.zero_grad()
    val = Feeder(
        lambda: np.full((2,255), 0.2, dtype=np.float32), 
        shape=[2,255]
    )
    res = nn(val)
    res.backward() 
    opt.step()

# In future release pass the idx
benchmark(lambda: ir_for(range(0, 1000), f), name="Tracking nodes")

Receiver(save_params, opt.parameters, name="saving params")

benchmark(lambda: execute(), name="full exec")
