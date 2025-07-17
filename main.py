from autodiff import Tensor, execute
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder

nn = TransformerEncoder(
    num_layers=4,
    d_model=128, 
    num_heads=4, 
    ff_dim=512
)

inp = Tensor.randn((4, 128))
res = nn(inp)
# res.backward()

# execute
execute()

# print_graph()