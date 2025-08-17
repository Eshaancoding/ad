from autodiff import execute, ir_for, Feeder, Receiver, pg
from autodiff.nn import SGD 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
import numpy as np

nn = MultiHeadAttention(
    d_model=512,
    num_heads=4
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
    res = nn(val, val, val)

    #res.backward() 
    #opt.step()

    Receiver(print_inp, [res], name="Printing res")

# In future release pass the idx
benchmark(lambda: ir_for(range(0, 100), f), name="Tracking nodes")

#Receiver(save_params, opt.parameters, name="saving params")

benchmark(lambda: execute(), name="full exec")
