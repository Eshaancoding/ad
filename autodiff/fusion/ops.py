from .base import FuseBase, Node, FuseType
import math
from ..graph import UnaryNode, BinaryNode, DotProdNode, ReduceNode

"""
Fuse ELW expressions together
"""

def is_elw (n: Node | FuseBase):
    return isinstance(n, UnaryNode) or isinstance(n, BinaryNode) or isinstance(n, ElwFuse)

def calc_size (n):
    if isinstance(n, FuseBase):
        return calc_size(n.nodes[-1])
    elif isinstance(n, Node):
        return math.prod(n.shape)

class ElwFuse (FuseBase):
    def __init__(self):
        super().__init__(FuseType.ALL)
        
    @staticmethod
    def could_fuse (node_one: Node | FuseBase, node_two: Node | FuseBase):
        return is_elw(node_one) and is_elw(node_two) and calc_size(node_one) == calc_size(node_two)
    
    # Short helper functions to find nodes easily
    def get_elw (self) -> UnaryNode | BinaryNode:
        for n in self.nodes:
            if isinstance(n, UnaryNode) or isinstance(n, BinaryNode):
                return n
        raise Exception("Unary or Binary node not found in ELW")
    
class DPElwFuse (FuseBase):
    def __init__(self):
        super().__init__(FuseType.ACROSS_LAYER)
        
    @staticmethod
    def could_fuse (node_one: Node | FuseBase, node_two: Node | FuseBase):
        return \
            (isinstance(node_one, DotProdNode) or isinstance(node_one, DPElwFuse)) and \
            is_elw(node_two) and \
            calc_size(node_one) == calc_size(node_two)
    
    # Short helper functions that find dp node easily
    def get_dp (self) -> DotProdNode:
        for n in self.nodes:
            if isinstance(n, DotProdNode):
                return n
        raise Exception("DotProd node not found")
    
class ReduceElwFuse (FuseBase):
    def __init__(self):
        super().__init__(FuseType.ACROSS_LAYER)
        
    @staticmethod
    def could_fuse (node_one: Node | FuseBase, node_two: Node | FuseBase):
        return \
            (isinstance(node_one, ReduceNode) or isinstance(node_one, ReduceElwFuse)) and \
            is_elw(node_two) and \
            calc_size(node_one) == calc_size(node_two)
            
    # Short helper functions that find dp node easily
    def get_reduce (self) -> ReduceNode:
        for n in self.nodes:
            if isinstance(n, ReduceNode):
                return n
        raise Exception("ReduceNode not found")
    
class Procedure (FuseBase):
    def __init__ (self):
        super().__init__(FuseType.ALL, is_proc=True)
        
    @staticmethod
    def could_fuse (_, _f):
        return True
