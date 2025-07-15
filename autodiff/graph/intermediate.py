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
    
    bank = []
    for node in nodes:
        # Walk through node. 
        def replace_node_with_bank (node: Node):
            for b in bank:
                if node.obj_eq(b):
                    temp_id = context.get_temp_id()
                    b.inter_out = temp_id
                    return IntermediateNode(temp_id, b.shape, b.res_expr)
           
        node.walk(replace_node_with_bank) 
    
        # Add to bank
        def fill_bank (node: Node):
            if isinstance(node, DotProdNode) or \
                isinstance(node, ReduceNode) or \
                isinstance(node, BinaryNode) or \
                isinstance(node, UnaryNode) or \
                isinstance(node, ContigiousNode):
            
                bank.append(node)
                
        node.walk(fill_bank)
     
    return nodes 