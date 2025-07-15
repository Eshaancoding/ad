from autodiff import Tensor, dot, proc, ir_for, execute, context
from autodiff.graph.opt import opt_node
from autodiff.expr import Global

a = Tensor.randn([4,8])
a_bias = Tensor.randn([8])
b = Tensor.randn([8,16])
b_bias = Tensor.randn([16])

inp = Tensor.randn((2, 4))

res = dot(inp, a) + a_bias
res = dot(res, b) + b_bias

res.backward()

a = a + a.grad
a_bias = a_bias + a_bias.grad
b = b + b.grad
b_bias = b_bias + b_bias.grad

# execute
execute()

# print the procedure
print(proc())