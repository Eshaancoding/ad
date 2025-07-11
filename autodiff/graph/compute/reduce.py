from ...node import Node
from ..helper import indent
from enum import Enum
from copy import deepcopy

class ReduceOp (Enum):
    SUM=1
    MAX=2
    
class ReduceNode (Node):
    def __init__(self, child:Node, op: ReduceOp):
        super().__init__([child])
        self.op = op
        
    def bck (self, grad:Node):
        repeat_n = self.child().shape()[-1]
        self.child().bck(grad.unsqueeze(-1).broadcast(-1, repeat_n))
        
    def shape (self):
        c = deepcopy(self.child().shape())
        del c[-1]
        return c
    
    def __repr__(self):
        return f"{self.op} on dim: -1\n{indent(self.child().__repr__())}"