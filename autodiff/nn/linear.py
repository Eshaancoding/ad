from ..graph.tensor import Tensor
from ..node import Node
from .module import Module

class Linear (Module):
    def __init__ (self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        
        self.w = Tensor.randn([input_dim, output_dim])
        self.b = Tensor.randn([output_dim]) if bias else None
        
    def forward (self, x: Node) -> Node:
        res = x @ self.w
        if self.b is not None: res = res + self.b
        return res