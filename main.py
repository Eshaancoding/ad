from copy import deepcopy
import pprint
from autodiff import Tensor, execute, ir_for, context, Feeder, Receiver
from autodiff.graph.compute.binary import BinaryNode, BinaryOp
from autodiff.graph.data.constant import ConstantNode
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD, linear 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
import numpy as np
from autodiff.print_graph import pg

class TestNeuralNet (Module):
    def __init__(self):
        super().__init__()

        self.l1 = Linear(512, 512, bias=False) 
        self.l2 = Linear(512, 512, bias=False)
        self.l3 = Linear(512, 512, bias=False)

        # later, compare with l1, l2, l3, etc.

    def forward(self, x:Node) -> Node:
        x = self.l1(x)
        x = self.l2(x) + 3*x.sigmoid()
        x = self.l3(x)
        return x
if True:
    nn = TransformerEncoder(
        num_layers=5, 
        d_model=512,
        num_heads=4,
        ff_dim=1024
    )

    #nn = TestNeuralNet()
else:
    nn = Sequential(
        Linear(512, 512, bias=False),
        Sigmoid(),
        Linear(512, 512)
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
    val = Feeder(lambda: get_inp(), shape=[10,512])
    res = nn(val)
    res.backward() 
    opt.step()

    Receiver(print_inp, [res], name="Printing res")

# In future release pass the idx
benchmark(lambda: ir_for(range(0, 100), f), name="Tracking nodes")

#Receiver(save_params, opt.parameters, name="saving params")

benchmark(lambda: execute(), name="full exec")
