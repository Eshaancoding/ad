######## Helpful for debugging! ######## 
# Finds the repeated intermediate operations between graphs within the same block
# ideally, you don't need this!
from typing import List
from ..node import Node
from colored import stylize, fore

from .compute.dotprod import DotProdNode
from .compute.reduce import ReduceNode
from .compute.binary import BinaryNode
from .compute.unary import UnaryNode
from .data.contigious import ContigiousNode
from ..context import context

from ..expr import NoneExpr

# not used in actual compilation. Only called during print_graph
class IntermediateNode (Node): 
    __match_args__ = ("node_id", "shape")
    def __init__(self, node_id: int, shape: list[int]):
        super().__init__([], phantom_shape=None)
        
        self.shape = shape
        self.node_id = node_id 
        self.res_expr = NoneExpr()
        
    def bck (self, grad:Node):
        raise Exception("Backward on an intermediate node!")
    
    def __repr__(self):
        return f"{stylize(f"Intermediate {self.node_id}", fore("cyan"))} with expr: {self.res_expr}"
    
    def format_single (self):
        return f"{self.id} = {stylize(f"Intermediate {self.node_id}", fore("cyan"))} with expr: {self.res_expr}"

def opt_intermediate (nodes: List[Node]):
    
   # Replace nodes with intermediates according to ID
    bank = {}
    new_nodes = []
    for node in nodes:
        # Walk through node. 
        def replace_node_with_bank (node: Node):
            if node.id in bank:
                b = bank[node.id]
                b.temp_id = b.id
                n = IntermediateNode(b.id, b.shape)
                n.res_expr = b.res_expr
                return n
                
            return node
           
        new_node = node.walk(replace_node_with_bank) 
    
        # Add to bank
        def fill_bank (node: Node):
            if isinstance(node, DotProdNode) or \
                isinstance(node, ReduceNode) or \
                isinstance(node, BinaryNode) or \
                isinstance(node, UnaryNode) or \
                isinstance(node, ContigiousNode):
            
                bank[node.id] = node
                
            return node
                
        node.walk(fill_bank)

        new_nodes.append(new_node)
    nodes = new_nodes
    
    # If not, well then
     
    return nodes 