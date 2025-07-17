from ...node import Node
from ..helper import indent
from enum import Enum
from copy import deepcopy
from ...expr import NoneExpr
from colored import stylize, fore, style

class ReduceOp (Enum):
    SUM=1
    MAX=2
    
class ReduceNode (Node):
    __match_args__ = ("child", "op")
    def __init__(self, child:Node, op: ReduceOp):
        super().__init__([child])
        self.child = child
        self.op = op
        assert len(child.shape) == 2, "Reduce shape must be 2-dim"

        self.res_expr = NoneExpr()

        # calc shape
        c = deepcopy(self.child.shape)
        del c[-1]
        self.shape = c
        
    def bck (self, grad:Node):
        repeat_n = self.child.shape[-1]
        self.child.bck(grad.unsqueeze(-1).broadcast(-1, repeat_n))
        
    def __repr__ (self):
        sh = self.child.shape
        x_dim = sh[0] 
        y_dim = sh[1]
        return f"{stylize(f"{self.temp_id} <-- ", fore("cyan")) if self.temp_id is not None else f"{self.id} = "}{self.op} on dim: -1 {stylize(f"(Vec/X: {x_dim}, Reduce/Y: {y_dim})", fore("yellow") + style("bold"))} --> {self.res_expr} ({self.child.id})"