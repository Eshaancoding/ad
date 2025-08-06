from autodiff import Tensor, execute, pg, concat, ir_for
from autodiff.graph.data.constant import ConstantNode
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD
from autodiff.nn.normalization.layernorm import LayerNorm
from autodiff.nn.transformer import TransformerEncoder
from autodiff.print_graph import pg

x = Tensor.randn([5,2])
y = Tensor.randn([1,2])

res = x * y + 2
res = res.T()

res_two = res * 3

res.keep()
res_two.keep()
execute()

print(res.val)
