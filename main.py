from autodiff import Tensor, execute, print_graph, concat, ir_for
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD
from autodiff.nn.transformer import TransformerEncoder

nn = TransformerEncoder(
    num_layers=1,
    d_model=128,
    num_heads=4,
    ff_dim=512
)

# nn = Sequential(
#     Linear(128, 512),
#     Sigmoid(),
#     Linear(512, 256),
#     Sigmoid()
# )

# opt = SGD(nn.parameters(), lr=0.1)

inp = Tensor.randn((4, 128))

def f ():
    res = nn(inp)
    # res.backward()
    # opt.step()

# In future release pass the idx
ir_for(range(0, 1000), f)

#
#
#
#
# execute
execute()

# print()
# print("===================== GRAPH! =====================")
