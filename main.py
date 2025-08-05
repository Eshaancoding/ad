from autodiff import Tensor, execute, ir_for, context
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD 
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark

if True:
    """
    nn = MultiHeadAttention(
        d_model=512,
        num_heads=4
    )
    """

    nn = TransformerEncoder(
        num_layers=2, # past 1 layer and it breaks pretty much; be careful of 100% core CPU util
        d_model=64,
        num_heads=4,
        ff_dim=128
    )
else:
    nn = Sequential(
        Linear(512, 256),
        Sigmoid(),
        Linear(256, 128),
        Sigmoid(),
    )

inp = Tensor.randn((2, 64))
opt = SGD(nn.parameters(), lr=0.01)
def f():
    res = nn(inp)
    res.keep()
    #res.backward() 
    #opt.step()

# In future release pass the idx
benchmark(lambda: ir_for(range(0, 1000), f), name="Tracking nodes")
benchmark(lambda: execute(), name="full exec")
