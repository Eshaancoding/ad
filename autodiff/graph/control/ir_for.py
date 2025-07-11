from ...node import Node
from ...context import Block
from ..helper import indent

#########################
## For node
class ForNode (Node):
    def __init__(self, block: Block, r: range):
        super().__init__([])
        
        self.block = block
        self.r = r
        
    def bck(self, grad):
        raise TypeError("Calling backward on a for node") 
    
    def shape (self):
        raise TypeError("Calling shape on a for node")
    
    def __repr__(self):
        return f"For {self.r.start} to {self.r.stop}:\n{indent(self.block.__repr__(), 2)}"