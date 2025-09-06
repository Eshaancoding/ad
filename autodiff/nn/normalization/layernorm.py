from ..module import Module
from ...graph.tensor import Tensor

class LayerNorm (Module):
    def __init__ (self, size: int):
        super().__init__()        

        self.scale = Tensor.ones([size], requires_grad=True)
        self.bias = Tensor.zeros([size], requires_grad=True)
        
    def forward (self, x):
        out_mean = x.mean(-1).unsqueeze(-1)
        variance = x.var(-1, 0).unsqueeze(-1)
        output = (x - out_mean) / (variance + 1e-5).sqrt()
        return output * self.scale + self.bias
