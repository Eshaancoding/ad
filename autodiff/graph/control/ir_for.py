from ...node import Node
from ...context import Block
from ..print import indent

#########################
## For node
class ForNode (Node):
    def __init__(self, block: Block, r: range):
        super().__init__([])
        
        # must define self.block in order to be treated as a "block node"
        self.block = block

        self.r = r
        
    def bck(self, grad):
        raise TypeError("Calling backward on a for node") 
    
    def __repr__(self):
        return f"For {self.r.start} to {self.r.stop}:\n{indent(self.block.__repr__(), 2)}"