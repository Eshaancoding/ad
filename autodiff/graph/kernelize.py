from copy import deepcopy
from ..node import Node
from .helper import global_to_ndim, ndim_to_global
from ..expr import *
from ..expr.simplify import simplify_expr
from enum import Enum
from typing import Dict
from .print import print_graph

from ..tensor import Tensor
from .compute.dotprod import DotProdNode
from .compute.reduce import ReduceNode
from .compute.binary import BinaryNode
from .compute.unary import UnaryNode
from .data.contigious import ContigiousNode
from .data.constant import ConstantNode
from .data import *

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

def fill_access_expr (node: Node, access_type: AccessType, visited: Dict[int, int] = {}):
    if node.id in visited:
        return

    match node:
        case DotProdNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            visited[node.id] = 1
            fill_access_expr(n.left, AccessType.XY, visited)
            fill_access_expr(n.right, AccessType.XY, visited)
        case ReduceNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            visited[node.id] = 1
            fill_access_expr(n.child, AccessType.XY, visited)
        case BinaryNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            visited[node.id] = 1
            fill_access_expr(n.left, AccessType.GLOBAL, visited)
            fill_access_expr(n.right, AccessType.GLOBAL, visited)
        case UnaryNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            visited[node.id] = 1
            fill_access_expr(n.child, AccessType.GLOBAL, visited)
        case ContigiousNode() as n:
            n.res_expr = make_access_type(access_type, n.shape)
            visited[node.id] = 1
            fill_access_expr(n.child, AccessType.GLOBAL, visited)
        case Tensor() as n:
            n.res_expr = make_access_type(access_type, n.shape) 
        case ConstantNode() as n:
            pass
        case _ as n:
            # TODO: Convert this function to walk?
            if hasattr(n, "left"): 
                fill_access_expr(n.left, access_type)
                visited[n.left.id] = 1
                fill_access_expr(n.right, access_type)
                visited[n.right.id] = 1
            else:
                fill_access_expr(n.child, access_type)
                visited[n.child.id] = 1

def simplify_data_cmds (node: Node) -> Node:
    # TODO: Fix constant simplification!!
    if isinstance(node, ConstantNode):
        return node
    for child in node.children():
        if isinstance(child, ConstantNode):
            return node

    match node:
        case PermuteNode(_ as child, _ as permute_to):
            child = simplify_data_cmds(child)
            res_expr = child.res_expr            

            new_res_expr = [Val(Constant(0)) for _ in range(len(res_expr))]
            dim = [0 for _ in range(len(res_expr))] 
            for i in range(len(res_expr)):
                new_res_expr[i] = deepcopy(res_expr[permute_to[i]])
                dim[i] = child.shape[permute_to[i]]
            
            child.res_expr = new_res_expr 
            child.shape = dim
            return child
        
        case ViewNode(_ as child, _ as target_dim):
            child = simplify_data_cmds(child)
            res_expr = child.res_expr            
            orig_dim = child.shape

            # in the rare case where we accessing [X,Y] and our shape is only one dim. Then... use only Y
            if len(orig_dim) == 1 and len(res_expr) == 2:
                res_expr = [res_expr[-1]]

            new_res_expr = global_to_ndim(ndim_to_global(res_expr, orig_dim), target_dim)
            child.res_expr = new_res_expr 
            child.shape = target_dim
            return child    

        case BroadcastNode(_ as child, _ as dim, _ as size):
            child = simplify_data_cmds(child)
            res_expr = child.res_expr            

            res_expr[dim] = Val(Constant(0))
            child.res_expr = res_expr
            child.shape[dim] = size
            return child
        
        case IndexNode(_ as child, _ as start, _ as end, _ as dim):
            child = simplify_data_cmds(child)
            res_expr = child.res_expr            

            res_expr[dim] = Add(res_expr[dim], Val(Constant(start)))
            child.res_expr = res_expr
            child.shape[dim] = end - start # update shape
            return child

        case _: 
            pass
        
    return node
            

def kernalize_node (node: Node) -> Node:
    fill_access_expr(node, AccessType.GLOBAL)

    # print_graph(node)    

    ##################################################
    # Simplify data cmds by altering access expressions
    node = node.walk(simplify_data_cmds, visited={})
    
    ##################################################
    # Simplify all exprs + globalize res expr
    def simplify_all_exprs (n:Node):
        if hasattr(n, "res_expr") and type(n.res_expr) == list:
            n.res_expr = simplify_expr(ndim_to_global(n.res_expr, n.shape))

        return n
            
    node = node.walk(simplify_all_exprs, visited={})
    
    return node