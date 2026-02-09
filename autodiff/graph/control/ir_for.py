from ...node import Node
from ...context import Block

#########################
## For node
class ForNode (Node):
    def __init__(self, block: Block, r: range):
        super().__init__([], [])
        
        # must define self.block in order to be treated as a "block node"
        self.block = block
        self.r = r
        
    def _bck(self, grad):
        raise TypeError("Calling backward on a for node") 
    
    def __repr__(self):
        return f"{self.id} = For {self.r.start} to {self.r.stop}:"

    def repeat_helper(self, is_child):
        return (
            "For",
            self.id
        )
