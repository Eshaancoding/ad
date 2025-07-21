from ..module import Module
from ...graph.tensor import Tensor

# TODO: test this?
class LayerNorm (Module):
    def __init__ (self, size: int):
        super().__init__()        

        self.scale = Tensor.ones([size])
        self.bias = Tensor.zeros([size])
        
    def forward (self, x):
        eps = 1e-9
        y = (x - x.mean(-1).unsqueeze(-1)) / ((x.var(-1) + eps).sqrt().unsqueeze(-1))
        return y * self.scale + self.bias