from typing import List
from ..tensor import Tensor
from ..node import Node
from .module import Module

class Sequential (Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for a in args: 
            assert isinstance(a, Module), "not a seq module!"
           
        self.modules: List[Module] = args
        
    def __call__(self, x:Node) -> Node:
        for mod in self.modules:
            x = mod(x)
        return x