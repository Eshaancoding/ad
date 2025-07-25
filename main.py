from autodiff import Tensor, execute, print_graph, concat, ir_for
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD
from autodiff.nn.transformer import TransformerEncoder
from time import time

# nn = TransformerEncoder(
#     num_layers=1,
#     d_model=128,
#     num_heads=4,
#     ff_dim=512
# )


nn = Sequential(
    Linear(512, 256),
    Sigmoid(),
    Linear(256, 128),
    Sigmoid()
)

opt = SGD(nn.parameters(), lr=0.1)

inp = Tensor.randn((16, 512))

def f ():
    res = nn(inp)
    res.backward()
    opt.step()

# In future release pass the idx
ir_for(range(0, 10_000), f)

# execute
print("Start...")
start = time()
execute()
print("elapsed: ", time()-start)