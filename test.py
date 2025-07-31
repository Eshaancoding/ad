from autodiff import Tensor, execute, print_graph, concat, ir_for
from autodiff.graph.data.constant import ConstantNode
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD
from autodiff.nn.transformer import TransformerEncoder

x = ConstantNode(2.0, [1,2])
y = ConstantNode(3.0, [1,2])
z = x * y
z *= Tensor.randn([1,2])
z.keep()

execute()
