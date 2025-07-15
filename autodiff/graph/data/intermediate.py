from ...node import Node
from ...expr import *

class IntermediateNode (Node): 
    def __init__(self, node_id: int, shape: list[int]):
        super().__init__([], phantom_shape=None)
        
        self.shape = shape
        self.node_id = node_id 
        self.res_expr = NoneExpr()
        
    def bck (self, grad:Node):
        raise Exception("Backward on an intermediate node!")
    
    def __repr__(self):
        return f"{stylize(f"Intermediate {self.node_id}", fore("cyan"))} with expr: {self.res_expr}"