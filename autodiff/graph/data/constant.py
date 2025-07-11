from ...node import Node
from typing import List

class ConstantNode (Node):
    def __init__(self, constant:float, dim: List[int]):
        super().__init__([])
        self.constant = constant 
        self.dim = dim
        
    def bck (self, _):
        pass # no backward for constant
    
    def shape (self) -> List[int]:
        return self.dim
    
    def __repr__ (self) -> str:
        return f"Const(val: {self.constant}, dim: {self.dim})"