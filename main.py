from autodiff import Tensor, dot, proc, ir_for, execute, context
from autodiff.graph.opt import opt_node

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


# justset the last for less debug
#context.procedure[0].nodes = [context.procedure[0].nodes[-1]]

# execute
execute()

# print the procedure
print(proc())

from autodiff.graph.data.constant import ConstantNode
from autodiff.graph.data.broadcast import BroadcastNode
from autodiff.graph.data.view import ViewNode
print(opt_node(
    ViewNode(
        ViewNode(
            BroadcastNode(
                ConstantNode(0.0, [1, 2])
            , 0, 6),
            [12]
        ),
        [6,2]
    )
))