from copy import deepcopy
import pprint
from autodiff import Tensor, execute, ir_for, context, Feeder, Receiver, pg
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
import numpy as np

if True:
    """
    nn = MultiHeadAttention(
        d_model=512,
        num_heads=4
    )
    """

    nn = TransformerEncoder(
        num_layers=1, # past 1 layer and it breaks pretty much; be careful of 100% core CPU util
        d_model=512,
        num_heads=4,
        ff_dim=128
    )
else:
    nn = Sequential(
        Linear(512, 256),
        Sigmoid(),
        Linear(256, 128),
    )

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
def f():
    opt.zero_grad()
    val = Feeder(get_inp, shape=[2,512], name="Get input")
    res = nn(val)

    res.backward() 
    opt.step()

    #Receiver(print_inp, [res], name="Printing res")

# In future release pass the idx
benchmark(lambda: ir_for(range(0, 100), f), name="Tracking nodes")

#Receiver(save_params, opt.parameters, name="saving params")

benchmark(lambda: execute(), name="full exec")


