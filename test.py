from autodiff import Tensor, execute, pg, concat, ir_for
from autodiff.graph.data.constant import ConstantNode
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD
from autodiff.nn.normalization.layernorm import LayerNorm
from autodiff.nn.transformer import TransformerEncoder
from autodiff.print_graph import pg

mod = LayerNorm(256)
opt = SGD(mod.parameters())

x = Tensor.randn([2, 256])
res = mod(x)

pg(res)

res.keep()
res.backward()
opt.step()

execute()
