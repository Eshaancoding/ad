from ...node import Node
from ...expr import *

class IntermediateNode (Node): 
    def __init__(self, _id: int, shape: list[int], res_expr:Expression):
        super().__init__([], phantom_shape=None)
        
        self.shape = shape
        self.id = _id 
        self.res_expr = res_expr
        
    def bck (self, grad:Node):
        raise Exception("Backward on an intermediate node!")
    
    def __repr__(self):
        return f"{stylize(f"Intermediate {self.id}", fore("cyan"))} with expr: {self.res_expr}"