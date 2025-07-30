from autodiff import Tensor, execute, print_graph, concat, ir_for
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD
from autodiff.nn.transformer import TransformerEncoder

x = Tensor.randn([8,4]) 
y = x.exp2().log2()
z = x.exp2() + 3

y.keep()
z.keep()

execute()
