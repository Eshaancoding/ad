from autodiff import Tensor, execute, pg, concat, ir_for
from autodiff.graph.data.constant import ConstantNode
from autodiff.graph.data.receiver import Receiver
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD
from autodiff.nn.normalization.layernorm import LayerNorm
from autodiff.nn.transformer import TransformerEncoder
from autodiff.print_graph import pg

#x = Tensor.randn([5,2])
y = Tensor.fill(1, [1,2])
z = Tensor.fill(2, [1,2])

x = z + y # 3
x += 3.0  # 6
res = x + y # 6 + 1 = 7

Receiver(lambda res, y, z, x: print(res, y, z, x), [res, y, z, x])

execute()

#print(res.val)
#print(y.val)
#print(z.val)
