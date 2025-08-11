from autodiff import Tensor, execute, ir_for, context, Feeder
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
import numpy as np

context.lenient_dep = True

if False:
    """
    nn = MultiHeadAttention(
        d_model=512,
        num_heads=4
    )
    """

    nn = TransformerEncoder(
        num_layers=1, # past 1 layer and it breaks pretty much; be careful of 100% core CPU util
        d_model=64,
        num_heads=2,
        ff_dim=128
    )
else:
    nn = Sequential(
        Linear(512, 256),
        Sigmoid(),
        Linear(256, 128),
        Sigmoid()
    )

idx = 0
def get_inp ():
    global idx
    if (idx+1) % 10 == 0:
        print(f"Get inp Idx: {idx+1}")
    idx += 1
    return np.full((2,512), 1.0, dtype=np.float32)

opt = SGD(nn.parameters(), lr=0.01)
inp_randn = Tensor.randn([2, 512])
def f():
    opt.zero_grad()
    #val = Feeder(get_inp, shape=[2,512], name="Get input")
    val = inp_randn
    res = nn(val)
    #res.keep()
    res.backward() 
    opt.step()

# In future release pass the idx
benchmark(lambda: ir_for(range(0, 100), f), name="Tracking nodes")
benchmark(lambda: execute(), name="full exec")
