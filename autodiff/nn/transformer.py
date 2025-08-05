from .sequential import Node, Sequential
from .linear import Linear
from .normalization import LayerNorm
from .module import Module
from .. import dot, concat
from math import sqrt
from typing import Optional

class Attention (Module):
    def __init__ (self):
        super().__init__()
        
    def forward (self, Q: Node, K: Node, V: Node, d_model:int, mask: Optional[Node] = None) -> Node:
        inner = dot(Q, K.T()) / sqrt(d_model)
        if mask is not None:
            inner = inner * mask

        return dot(
            inner.softmax(-1),
            V
        )
    
class MultiHeadAttention (Module):
    def __init__ (self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num heads"
        d_v = d_model // num_heads
        
        self.wq = [Linear(d_model, d_v, bias=False) for _ in range(num_heads)]
        self.wk = [Linear(d_model, d_v, bias=False) for _ in range(num_heads)]
        self.wv = [Linear(d_model, d_v, bias=False) for _ in range(num_heads)]
        self.attentions = [Attention() for _ in range(num_heads)]
        self.wo = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model
        
    def forward (self, Q: Node, K: Node, V: Node, mask: Optional[Node] = None) -> Node:
        head_out = [
            self.attentions[i](
                self.wq[i](Q),
                self.wk[i](K),
                self.wv[i](V),
                self.d_model,
                mask
            )
            for i in range(self.num_heads)
        ]

        return self.wo(concat(head_out, -1))
    
class AttentionFeedforward (Module):
    def __init__ (self, d_model: int, inner_dim: int):
        super().__init__()
        
        self.f_expand = Linear(d_model, inner_dim)
        self.f_contract = Linear(inner_dim, d_model)
        
    def forward (self, x: Node):
        return self.f_contract(self.f_expand(x).relu())
    
class TransformerEncoderLayer (Module):
    def __init__ (self, d_model: int, num_heads: int, ff_dim: int):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_att = LayerNorm(d_model)
        self.layer_norm_ffwd = LayerNorm(d_model)
        self.ffwd = AttentionFeedforward(d_model, ff_dim)
        
    def forward (self, x: Node) -> Node:
        y = self.attention(x, x, x)
        y = self.layer_norm_att(x + y)     
        
        z = self.ffwd(y)
        #z = self.layer_norm_ffwd(y + z)
        
        return z
    
class TransformerEncoder (Module):
    def __init__ (self, num_layers: int, d_model: int, num_heads: int, ff_dim: int):
        super().__init__()
        
        self.seq = Sequential(*[
            TransformerEncoderLayer(d_model, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
    def forward (self, x) -> Node:
        return self.seq(x)
