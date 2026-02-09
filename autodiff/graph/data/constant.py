from ...node import Node
from typing import List

class ConstantNode (Node):
    __match_args__ = ("constant", "shape")
    def __init__(self, constant:float, shape: List[int]):
        super().__init__([], shape)
        self.constant = constant 
        
    def _bck (self, _):
        return None
    
    def __repr__ (self) -> str:
        return f"{self.id} = Const(val: {self.constant}, dim: {self.shape})"

    def repeat_helper (self, is_child):
        return (self.constant, tuple(self.shape))
