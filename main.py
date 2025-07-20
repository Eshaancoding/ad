from autodiff import Tensor, execute, print_graph
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder

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

inp = Tensor.randn((4, 128))
res = nn(inp)
# res.backward()

# execute
execute()

# print_graph()