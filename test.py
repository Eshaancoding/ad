from autodiff import Tensor, execute, print_graph, concat, ir_for
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD
from autodiff.nn.transformer import TransformerEncoder

x = Tensor.randn([4,8])
res = (x.sum(0)) * 3.0

execute()