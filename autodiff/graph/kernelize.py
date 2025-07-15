from copy import deepcopy
from ..node import Node
from .helper import replace_patterns, global_to_ndim, ndim_to_global
from ..expr import *
from ..expr.simplify import simplify_expr
from enum import Enum

from ..tensor import Tensor
from .compute.dotprod import DotProdNode
from .compute.reduce import ReduceNode
from .compute.binary import BinaryNode
from .compute.unary import UnaryNode
from .data.contigious import ContigiousNode
from .data.constant import ConstantNode
from .data.intermediate import IntermediateNode

from ..phantom import *

class AccessType (Enum):
    GLOBAL=1 # for elw/unary
    XY=2     # for reduce and dot prod 

def make_access_type (access_type: AccessType, shape: list[int]): 
    match access_type:
        case AccessType.XY:
            return [X(), Y()]
        case AccessType.GLOBAL:
            return global_to_ndim(Global(), shape)

def fill_access_expr (node: Node, access_type: AccessType):
    match node:
        case DotProdNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            fill_access_expr(n.left(), AccessType.XY)
            fill_access_expr(n.right(), AccessType.XY)
        case ReduceNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            fill_access_expr(n.child(), AccessType.XY)
        case BinaryNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            fill_access_expr(n.left(), AccessType.GLOBAL)
            fill_access_expr(n.right(), AccessType.GLOBAL)
        case UnaryNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            fill_access_expr(n.child(), AccessType.GLOBAL)
        case ContigiousNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            fill_access_expr(n.child(), AccessType.GLOBAL)
        case Tensor() as n:
            n.res_expr = make_access_type(access_type, n.shape) 
        case ConstantNode() as n:
            pass
        case IntermediateNode() as n:
            pass
        case _ as n:
            if len(n.children) == 2:
                fill_access_expr(n.left(), access_type)
                fill_access_expr(n.right(), access_type)
            else:
                fill_access_expr(n.child(), access_type)

def kernalize_node (node: Node) -> Node:
    fill_access_expr(node, AccessType.GLOBAL)

    ##################################################
    # Simplify permute
    def simplify_permute (n:Node):
        try:
            res_expr = n.child().res_expr
        except Exception:
            raise Exception("Can't access expr at permute child")
        
        new_res_expr = [Val(Constant(0)) for _ in range(len(res_expr))]
        dim = [0 for _ in range(len(res_expr))] 
        for i in range(len(res_expr)):
            new_res_expr[i] = deepcopy(res_expr[n.permute_to[i]])
            dim[i] = n.child().shape[n.permute_to[i]]
        
        new_node = deepcopy(n.child())
        new_node.res_expr = new_res_expr 
        new_node.shape = dim
        return new_node
    
    ##################################################
    # Simplify view
    def simplify_view (n:Node):
        try:
            res_expr = n.child().res_expr
        except Exception:
            raise Exception("Can't access expr at view child")
        
        target_dim = deepcopy(n.shape)        
        orig_dim = deepcopy(n.child().shape)

        new_res_expr = global_to_ndim(ndim_to_global(res_expr, orig_dim), target_dim)
        new_node = deepcopy(n.child())
        new_node.res_expr = new_res_expr 
        new_node.shape = target_dim
        return new_node

    ##################################################
    # Simplify broadcast
    def simplify_broadcast (n:Node):
        try:
            res_expr = deepcopy(n.child().res_expr)
        except:
            print(n)
            raise Exception("Can't access expr at broadcast child")
        
        res_expr[n.dim] = Val(Constant(0))
        new_node = deepcopy(n.child())
        new_node.res_expr = res_expr
        new_node.shape[n.dim] = n.size
        return new_node
    
    ##################################################
    # Simplify index
    def simplify_index (n:Node):
        try:
            res_expr = deepcopy(n.child().res_expr)
        except Exception:
            raise Exception("Can't access expr at index child")
        
        res_expr[n.dim] = Add(res_expr[n.dim], Val(Constant(n.start)))
        new_node = deepcopy(n.child())
        new_node.res_expr = res_expr
        new_node.shape[n.dim] = n.end - n.start # update shape
        return new_node

    node = replace_patterns(node, [
        (PhantomResultNode([2,4]).permute([1,0]), simplify_permute),
        (PhantomResultNode([2,4]).view([8]), simplify_view),
        (PhantomResultNode([1,4]).broadcast(0, 8), simplify_broadcast),
        (PhantomResultNode([2,4])[:,0], simplify_index)
    ]) 
    
    ##################################################
    # Simplify all exprs + globalize res expr
    def simplify_all_exprs (n:Node):
        from ..context import context
        if isinstance(n, IntermediateNode):
            n.res_expr = context.temp_to_expr[n.node_id]
        elif hasattr(n, "res_expr"):
            n.res_expr = simplify_expr(ndim_to_global(n.res_expr, n.shape))
            if n.inter_out is not None:
                context.temp_to_expr[n.inter_out] = n.res_expr
            
            
        for child in n.children:
            simplify_all_exprs(child)

    simplify_all_exprs(node)
    
    return node