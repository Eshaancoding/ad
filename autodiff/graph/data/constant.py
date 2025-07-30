from ...node import Node
from typing import List

class ConstantNode (Node):
    __match_args__ = ("constant", "shape")
    def __init__(self, constant:float, shape: List[int]):
        super().__init__([], shape)
        self.constant = constant 
        
    def bck (self, _):
        pass # no backward for constant
    
    def __repr__ (self) -> str:
        return f"{self.id} = Const(val: {self.constant}, dim: {self.shape})"

    def node_eq(self, other) -> bool:
        if not isinstance(other, ConstantNode):
            return False
        
        return self.constant == other.constant and \
               self.shape == other.shape
