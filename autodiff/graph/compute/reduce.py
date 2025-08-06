from ...node import Node
from enum import Enum
from copy import deepcopy
from colored import stylize, fore, style

class ReduceOp (Enum):
    SUM=1
    MAX=2
    
class ReduceNode (Node):
    __match_args__ = ("child", "op")
    def __init__(self, child:Node, op: ReduceOp):
        assert len(child.shape) == 2, "Reduce shape must be 2-dim"
        
        # Calc shape and self.op
        res_shape = deepcopy(child.shape)
        del res_shape[-1]

        # initialize parent node
        super().__init__([child], res_shape)
        self.op = op
        
    def bck (self, grad:Node):
        repeat_n = self.children_shapes[0][-1]
        self.child.bck(grad.unsqueeze(-1).broadcast(-1, repeat_n))
        
    def __repr__ (self):
        sh = self.children_shapes[0]
        x_dim = sh[0] 
        y_dim = sh[1]
        size_str = stylize(f"(Vec/X: {x_dim}, Reduce/Y: {y_dim})", fore("yellow") + style("bold"))
        
        if self.kargs[0].is_none() or self.kres.is_none():
            return f"{self.id} = {self.op} on dim: -1 {size_str} --> ({self.child.id})"
        else:
            return f"{self.kres} = {self.op} on dim: -1 {size_str} --> ({self.kargs[0]})"

    def node_eq(self, other) -> bool:
        if not isinstance(other, ReduceNode):
            return False

        return \
            self.op == other.op and \
            self.child.node_eq(other.child)
