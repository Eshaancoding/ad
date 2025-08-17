import math
from autodiff.fusion.base import FuseBase
from ..graph import UnaryNode, BinaryNode, DotProdNode, ReduceNode, Node

def is_elw (node: Node):
    return isinstance(node, UnaryNode) or isinstance(node, BinaryNode)

def calc_size (n):
    if isinstance(n, FuseBase):
        return calc_size(n.nodes[-1])
    elif isinstance(n, Node):
        return math.prod(n.shape)

class DPElwFuse (FuseBase):
    def __init__ (self):
        super().__init__()

    def _fuse (self, node_one, node_two):
        return \
            (isinstance(node_one, DotProdNode) or is_elw(node_one)) and \
            (is_elw(node_two)) and \
            calc_size(node_one) == calc_size(node_two)
    
    def _init (self, node):
        return isinstance(node, DotProdNode)

class ReduceElwFuse (FuseBase):
    def __init__ (self):
        super().__init__()

    def _fuse (self, node_one, node_two):
        return \
            (isinstance(node_one, ReduceNode) or is_elw(node_one)) and \
            (is_elw(node_two)) and \
            calc_size(node_one) == calc_size(node_two)
    
    def _init (self, node):
        return isinstance(node, ReduceNode)

class ElwFuse (FuseBase):
    def __init__ (self):
        super().__init__(dep_match=False)

    def _fuse (self, node_one, node_two):
        return \
            (is_elw(node_one) and is_elw(node_two)) and \
            calc_size(node_one) == calc_size(node_two)
    
    def _init (self, node):
        return is_elw(node)
