from copy import deepcopy
from autodiff import Tensor, execute, ir_for, context, Feeder, Receiver
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
import numpy as np

nn = Sequential(
    Linear(512, 256),
    Sigmoid(),
    Linear(256, 16)
)

def get_inp ():
    return np.full((2,512), 1.0, dtype=np.float32)

def save_params (param_one, param_two, param_three, param_four):
    print(param_one.shape)
    print(param_two.shape)
    print(param_three.shape)
    print(param_four.shape)

opt = SGD(nn.parameters(), lr=0.01)
def f():
    opt.zero_grad()
    val = Feeder(get_inp, shape=[2,512], name="Get input")
    res = nn(val)
    res.backward() 
    opt.step()

    #Receiver(print_inp, [res], name="Printing res")
benchmark(lambda: ir_for(range(0, 100), f), name="Tracking nodes")

# In future release pass the idx
Receiver(save_params, opt.parameters, name="saving params")

benchmark(lambda: execute(), name="full exec")
