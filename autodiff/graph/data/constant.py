from ...node import Node
from typing import List

class ConstantNode (Node):
    def __init__(self, constant:float, dim: List[int]):
        super().__init__([])
        self.constant = constant 
        self.shape = dim
        
    def bck (self, _):
        pass # no backward for constant
    
    def __repr__ (self) -> str:
        return f"Const(val: {self.constant}, dim: {self.shape})"