from ...node import Node
from ...expr import NoneExpr
from colored import stylize, fore

# This is a light wrapper over the node. However, it's pretty important for optimizations
# and is used extensively when converting to Kernel procedure
class ContigiousNode (Node):
    __match_args__ = ("child")
    def __init__(self, child:Node):
        super().__init__([child])

        self.child = child
        self.res_expr = NoneExpr()
        self.shape = self.child.shape
        
    def bck(self, grad:Node):
        self.child.bck(grad)
        
    def __repr__ (self):
        return f"{self.id} = Contigious --> {self.res_expr} ({self.child.id})"