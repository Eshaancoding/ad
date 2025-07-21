from autodiff import Tensor, execute, print_graph, concat
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder

a = Tensor.randn([4,8])
b = Tensor.randn([8,4])
c = Tensor.randn([32])
res = concat([a.T(), b, c.view([8,4])], 1)
res = res.T()
res = res * 3

# nn = TransformerEncoder(
#     num_layers=1,
#     d_model=128, 
#     num_heads=4, 
#     ff_dim=512
# )

# nn = Sequential(
#     Linear(128, 512),
#     Sigmoid(),
#     Linear(512, 256),
#     Sigmoid()
# )

# inp = Tensor.randn((4, 128))
# res = nn(inp)
# res.backward()

# execute
execute()


print()
print("===================== GRAPH! =====================")
print_graph()