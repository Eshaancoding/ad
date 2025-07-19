from ..expr import *
from ..expr.simplify import simplify_expr
from .helper import global_to_ndim, ndim_to_global, walk_graph, ndim_change_datacmds
from ..node import Node
from ..context import Context
from .print import print_graph

from .compute import *
from .data import *

from typing import Optional, Dict, Tuple
from enum import Enum

class AccessType (Enum):
    GLOBAL=1
    XY=2
    
def pot_child_datacmd (node: Node) -> Tuple[Node, Optional[Node], bool]:
    if len(node.children()) == 0:
        return (node, None, False)

    match node:
        case PermuteNode() as n:
            return (node.child, n, True)
        case ViewNode() as n:
            return (node.child, n, True)
        case BroadcastNode() as n:
            return (node.child, n, True)
        case IndexNode() as n:
            return (node.child, n, True)
     
    return (node, None, False)
    
# TODO: Fix the while (true)... ugly code
def fill_child_datacmds (node: Node, _):
    match node:
        case DotProdNode() | BinaryNode() as n:
            # handle left node
            while True:
                n.left, cmd, did_repl = pot_child_datacmd(n.left)
                if isinstance(cmd, Node):
                    n.children_datacmds[0].append(cmd)
                if not did_repl:
                    break

            # handle right node
            while True:
                n.right, cmd, did_repl = pot_child_datacmd(n.right)
                if isinstance(cmd, Node):
                    n.children_datacmds[1].append(cmd)
                if not did_repl:
                    break
        case ReduceNode() | UnaryNode() | ContigiousNode() as n:
            # handle child node
            while True:
                n.child, cmd, did_repl = pot_child_datacmd(n.child)
                if isinstance(cmd, Node):
                    n.children_datacmds[0].append(cmd)
                if not did_repl:
                    break
                
def calc_exprs (node: Node, _):
    match node:
        case DotProdNode() as n:
            # calc left
            left_dim = [X(), Y()]
            left_dim = ndim_change_datacmds(left_dim, n.children_datacmds[0])
            n.children_exprs[0] = simplify_expr(ndim_to_global(left_dim, n.children_shapes[0]))
            
            # calc right
            right_dim = [X(), Y()]
            right_dim = ndim_change_datacmds(right_dim, n.children_datacmds[1])
            n.children_exprs[1] = simplify_expr(ndim_to_global(right_dim, n.children_shapes[1]))

            # remove data cmds
            n.children_datacmds = None

        case BinaryNode() as n:
            # calc left
            left_dim = global_to_ndim(Global(), n.left.shape) # NOTE: Using n.child, not n.children_shape. We *want* the original shape
            left_dim = ndim_change_datacmds(left_dim, n.children_datacmds[0])
            n.children_exprs[0] = simplify_expr(ndim_to_global(left_dim, n.children_shapes[0]))
            
            # calc right
            right_dim = global_to_ndim(Global(), n.right.shape) # NOTE: Using n.child, not n.children_shape. We *want* the original shape
            right_dim = ndim_change_datacmds(right_dim, n.children_datacmds[1])
            n.children_exprs[1] = simplify_expr(ndim_to_global(right_dim, n.children_shapes[1]))

            # remove data cmds
            n.children_datacmds = None

        case ReduceNode() as n:
            # calc child
            child_dim = [X(), Y()]
            child_dim = ndim_change_datacmds(child_dim, n.children_datacmds[0])
            n.children_exprs[0] = simplify_expr(ndim_to_global(child_dim, n.children_shapes[0]))
            
            # remove data cmds
            n.children_datacmds = None
        
        case UnaryNode() | ContigiousNode() as n:
            # calc child
            child_dim = global_to_ndim(Global(), n.child.shape) # NOTE: Using n.child, not n.children_shape. We *want* the original shape
            child_dim = ndim_change_datacmds(child_dim, n.children_datacmds[0])
            n.children_exprs[0] = simplify_expr(ndim_to_global(child_dim, n.children_shapes[0]))
            
            # remove data cmds
            n.children_datacmds = None
        
            
             
                
def kernalize (context: Context):
    for idx in range(len(context.procedure)):
        context.procedure[idx].nodes = walk_graph(context.procedure[idx].nodes, fill_child_datacmds)
        context.procedure[idx].nodes = walk_graph(context.procedure[idx].nodes, calc_exprs)
                
                