from ...node import Node
from ..module import Module

class Sigmoid (Module):
    def __init__ (self):
        super().__init__()
        
    def forward (self, x: Node) -> Node:
        return x.sigmoid()