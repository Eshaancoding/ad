from autodiff.graph.data.receiver import Receiver
from autodiff.graph.tensor import Tensor
from autodiff.helper import benchmark
from autodiff.nn.normalization.layernorm import LayerNorm
import autodiff
import torch

from autodiff.nn.transformer import Attention

def assert_results (output, wq_one, wk_one, wv_one):
    print("******** Output ********")
    print(output)  

    print()
    print("******** WQ One ********")
    print(wq_one.flatten()[0:10])

    print()
    print("******** WK One ********")
    print(wk_one.flatten()[0:10])

    print()
    print("******** WV One ********")
    print(wv_one.flatten()[0:10])

def test_transformer ():
    torch.manual_seed(42)

    # num heads = 2, num layers = 1
    d_model = 32
    inner_dim = 64
    d_v = 16 

    # initialize layer norm
    # TODO: not sure if implementation for layer norm is different
    layer_norm_att = LayerNorm(d_model)
    layer_norm_ffwd = LayerNorm(d_model)

    # initialize head one
    wq_one = Tensor.from_torch(torch.randn(d_model, d_v), True)
    wk_one = Tensor.from_torch(torch.randn(d_model, d_v), True)
    wv_one = Tensor.from_torch(torch.randn(d_model, d_v), True)

    # initialize head two
    wq_two = Tensor.from_torch(torch.randn(d_model, d_v), True)
    wk_two = Tensor.from_torch(torch.randn(d_model, d_v), True)
    wv_two = Tensor.from_torch(torch.randn(d_model, d_v), True)
    
    # initialize wo weight
    wo = Tensor.from_torch(torch.randn(d_model, d_model), True)

    # initialize feed forward 
    w_expand = Tensor.from_torch(torch.randn(d_model, inner_dim), True)
    w_contract = Tensor.from_torch(torch.randn(inner_dim, d_model), True)

    # initialize input
    input = Tensor.from_torch(torch.randn(10, d_model), True)
    
    # ******** Starting output ******** 
    # Attention
    y = [
        Attention().forward(input @ wq_one, input @ wk_one, input @ wv_one, d_model, None),
        Attention().forward(input @ wq_two, input @ wk_two, input @ wv_two, d_model, None)
    ]
    y = autodiff.concat(y, -1) @ wo
    y = layer_norm_att(input + y)
    
    z = (y @ w_expand).relu() @ w_contract
    z = layer_norm_ffwd(y + z)
    z.backward()

    print(z.shape)
    print(wq_one.grad().shape)
    print(wk_one.grad().shape)

    Receiver(assert_results, [
        z, 
        wq_one.grad(), 
        wk_one.grad(),
        wv_one.grad()
    ], name="getting params")

    benchmark(lambda: autodiff.execute(), name="full exec")

if __name__ == "__main__":
    test_transformer()
