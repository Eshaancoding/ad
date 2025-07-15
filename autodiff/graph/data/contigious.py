from ...node import Node
from ..helper import indent
from ...expr import NoneExpr
from colored import stylize, fore

# This is a light wrapper over the node. However, it's pretty important for optimizations
# and is used extensively when converting to Kernel procedure
class ContigiousNode (Node):
    def __init__(self, child:Node):
        super().__init__([child])

        self.res_expr = NoneExpr()
        self.shape = self.child().shape
        
    def bck(self, grad:Node):
        self.child().bck(grad)
        
    def __repr__ (self):
        return f"{stylize(f"{self.inter_out} <-- ", fore("cyan")) if self.inter_out is not None else ""}Contigious --> {self.res_expr}\n{indent(self.child().__repr__())}"