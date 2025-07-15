# Phantom nodes that's helpful for pattern recognition
from .node import Node

from .tensor import Tensor
from .graph.compute.dotprod import DotProdNode
from .graph.compute.reduce import ReduceNode
from .graph.compute.binary import BinaryNode
from .graph.compute.unary import UnaryNode
from .graph.data.contigious import ContigiousNode

class PhantomNode (Node):
    def __init__(self, shape):
        super().__init__([], phantom_shape=shape)

    def phantom_type_eq (self, other):
        return isinstance(other, Node)

# nodes that have a result expr 
# DotProd, Reduce, Binary, Unary, Contigious, Tensor all have result expr
class PhantomResultNode (PhantomNode):
    def __init__(self, shape):
        super().__init__(shape)
        
    def phantom_type_eq (self, other):
        res = isinstance(other, DotProdNode) \
                or isinstance(other, ReduceNode) \
                or isinstance(other, BinaryNode) \
                or isinstance(other, UnaryNode) \
                or isinstance(other, ContigiousNode) \
                or isinstance(other, Tensor)            
                
        return res