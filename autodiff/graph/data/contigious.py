from ...node import Node
from ..helper import indent

# This is a light wrapper over the node. However, it's pretty important for optimizations
# and is used extensively when converting to Kernel procedure
class ContigiousNode (Node):
    def __init__(self, child:Node):
        super().__init__([child])
        
    def bck(self, grad:Node):
        self.child().bck(grad)
        
    def shape (self):
        return self.child().shape()
    
    def __repr__ (self):
        return f"Contigious\n{indent(self.child().__repr__())}"