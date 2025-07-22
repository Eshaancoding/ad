from ...node import Node
from ...expr import NoneExpr
from colored import stylize, fore

# This is a light wrapper over the node. However, it's pretty important for optimizations
# and is used extensively when converting to Kernel procedure
class ContigiousNode (Node):
    __match_args__ = ("child")
    def __init__(self, child:Node):
        super().__init__([child], child.shape)
        
    def bck(self, grad:Node):
        self.child.bck(grad)
        
    def __repr__ (self):
        if self.kargs[0].is_none():
            return f"{self.id} = Contigious ({self.child.id}: {self.children_exprs[0]})"
        else:
            return f"{self.id} = Contigious ({self.child.id}: {self.kargs[0]})"