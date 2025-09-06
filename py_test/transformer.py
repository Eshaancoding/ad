import torch
import torch.nn as nn
import torch.nn.functional as F

def test_transformer():
    torch.manual_seed(42)

    # Model dimensions and config
    d_model = 32
    inner_dim = 64
    d_v = 16

    # LayerNorm modules
    layer_norm_att = nn.LayerNorm(d_model)
    layer_norm_ffwd = nn.LayerNorm(d_model)

    # Attention head weights
    wq_one = torch.randn(d_model, d_v, requires_grad=True)
    wk_one = torch.randn(d_model, d_v, requires_grad=True)
    wv_one = torch.randn(d_model, d_v, requires_grad=True)

    wq_two = torch.randn(d_model, d_v, requires_grad=True)
    wk_two = torch.randn(d_model, d_v, requires_grad=True)
    wv_two = torch.randn(d_model, d_v, requires_grad=True)

    # Output projection and ff weights
    wo = torch.randn(2 * d_v, d_model, requires_grad=True)  # concat two heads
    w_expand = torch.randn(d_model, inner_dim, requires_grad=True)
    w_contract = torch.randn(inner_dim, d_model, requires_grad=True)

    # Input
    x = torch.randn(10, d_model, requires_grad=True)

    # Attention for each head
    def attention(q, k, v, scale):
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        weights = torch.softmax(attn, dim=-1)
        return torch.matmul(weights, v)

    scale = d_model ** 0.5
    y1 = attention(x @ wq_one, x @ wk_one, x @ wv_one, scale)
    y2 = attention(x @ wq_two, x @ wk_two, x @ wv_two, scale)
    y_cat = torch.cat([y1, y2], dim=-1) @ wo

    # Add & Norm
    y_out = layer_norm_att(x + y_cat)

    # Feedforward + Norm
    z = F.relu(y_out @ w_expand) @ w_contract
    out = layer_norm_ffwd(y_out + z)

    # Backward and print grads for head one params (as example)
    out.sum().backward()

    print(f"input: {x.flatten()[0:5]}")

    print("Grad for wk_one:", wk_one.grad.flatten()[0:10])
    print("Grad for wv_one:", wv_one.grad.flatten()[0:10])
    print("Grad for x:", x.grad.flatten()[0:10])
    print("Output:", out)

if __name__ == "__main__":
    test_transformer()
