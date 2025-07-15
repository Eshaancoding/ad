# Finds the repeated intermediate operations between graphs within the same block
from typing import List
from ..node import Node

from .compute.dotprod import DotProdNode
from .compute.reduce import ReduceNode
from .compute.binary import BinaryNode
from .compute.unary import UnaryNode
from .data.contigious import ContigiousNode
from .data.intermediate import IntermediateNode

from ..context import context

def opt_intermediate (nodes: List[Node]):
    
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
                
        node.walk(fill_bank)

        new_nodes.append(new_node)
     
    return new_nodes 