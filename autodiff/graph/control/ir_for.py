from ...node import Node
from ...context import Block
from ...print import indent

#########################
## For node
class ForNode (Node):
    def __init__(self, block: Block, r: range):
        super().__init__([], [])
        
        # must define self.block in order to be treated as a "block node"
        self.block = block
        self.r = r
        
    def bck(self, grad):
        raise TypeError("Calling backward on a for node") 
    
    def __repr__(self):
        return f"{self.id} = For {self.r.start} to {self.r.stop}:"

    def node_eq(self, other) -> bool:
        raise Exception("For node on node_eq")
