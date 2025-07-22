from ..expr import *
from ..expr.simplify import simplify_expr
from ..helper import global_to_ndim, ndim_to_global, walk_graph, ndim_change_datacmds
from ..node import Node
from ..context import Context
from ..print import print_graph

from ..graph import *
from . import KernelArg, KMatrix, KConcat, KConstant

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
        case DotProdNode() | BinaryNode() | ConcatNode() as n:
            # handle left node
            while True:
                n.left, cmd, did_repl = pot_child_datacmd(n.left)
                if isinstance(cmd, Node):
                    n.children_datacmds[0].insert(0, cmd)
                if not did_repl:
                    break

            # handle right node
            while True:
                n.right, cmd, did_repl = pot_child_datacmd(n.right)
                if isinstance(cmd, Node):
                    n.children_datacmds[1].insert(0, cmd)
                if not did_repl:
                    break
        case ReduceNode() | UnaryNode() | ContigiousNode() as n:
            # handle child node
            while True:
                n.child, cmd, did_repl = pot_child_datacmd(n.child)
                if isinstance(cmd, Node):
                    n.children_datacmds[0].insert(0, cmd)
                if not did_repl:
                    break
               
def make_karg (initial_dim, child: Node, data_cmds, shape):
    if isinstance(child, ConstantNode):
        return KConstant(child.constant)
    elif isinstance(child, ConcatNode):
        idx_end = child.children_shapes[0][child.dim]

        # extend to the data cmds
        child.children_datacmds[0].extend(data_cmds)
        child.children_datacmds[1].extend(data_cmds)
        
        # for the second child, offset the dim by idx_end (we do this via IndexNode)
        child.children_datacmds[1].append(IndexNode(
            Node([], []), 
            start=-idx_end, 
            end=-1, 
            dim=child.dim, 
            for_concat=True
        )) 

        karg_one = make_karg(
            initial_dim,
            child.left,
            child.children_datacmds[0],
            child.children_shapes[0]
        )
        
        karg_two=make_karg(
            initial_dim,
            child.right,
            child.children_datacmds[1],
            child.children_shapes[1]
        )

        # Calculate the "access" dim. This is by taking the initial dim and going through the data cmds
        # Then apply this as condition
        # TODO: Check if this works lol
        if initial_dim == Global():
            initial_dim = global_to_ndim(initial_dim, child.shape)
        initial_dim = ndim_change_datacmds(initial_dim, data_cmds)
        condition = simplify_expr(MoreThan(initial_dim[child.dim], Val(Constant(idx_end-1))), None)
        
        return KConcat(
            karg_one,
            karg_two,
            condition=condition,
            shape=shape
        )
        
    else:
        if initial_dim == Global():
            initial_dim = global_to_ndim(initial_dim, child.shape)
        dim = ndim_change_datacmds(initial_dim, data_cmds)
        return KMatrix(child.id, dim, shape)
    
def simplify_karg (child: KernelArg, size: Optional[int]=None):
    if isinstance(child, KMatrix):
        child.access = simplify_expr(ndim_to_global(child.access, child.shape), size)
        # child.access = simplify_expr(child.access)
    elif isinstance(child, KConcat):
        simplify_karg(child.karg_one) 
        simplify_karg(child.karg_two)
    return child

def calc_exprs (node: Node, _):
    match node:
        case DotProdNode() as n:
            # calc left
            n.kargs[0] = simplify_karg(make_karg(
                initial_dim=[X(), Y()],
                child=n.left,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            ))

            n.kargs[1] = simplify_karg(make_karg(
                initial_dim=[X(), Y()],
                child=n.right,
                data_cmds=n.children_datacmds[1],
                shape=n.children_shapes[1]
            ))

            # remove data cmds
            n.children_datacmds = None

        case BinaryNode() as n:
            size = math.prod(n.shape) 

            # calc left
            n.kargs[0] = simplify_karg(make_karg(
                initial_dim=Global(),
                child=n.left,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            ), size)
            
            n.kargs[1] = simplify_karg(make_karg(
                initial_dim=Global(),
                child=n.right,
                data_cmds=n.children_datacmds[1],
                shape=n.children_shapes[1]
            ), size)

            # remove data cmds
            n.children_datacmds = None

        case ReduceNode() as n:
            # calc child
            n.kargs[0] = simplify_karg(make_karg(
                initial_dim=[X(), Y()],
                child=n.child,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            ))
            
            # remove data cmds
            n.children_datacmds = None
        
        case UnaryNode() | ContigiousNode() as n:
            size = math.prod(n.shape) 
            
            # calc child
            n.kargs[0] = simplify_karg(make_karg(
                initial_dim=Global(),
                child=n.child,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            ), size)

            # remove data cmds
            n.children_datacmds = None
        
def kernalize (context: Context):
    for idx in range(len(context.procedure)):
        context.procedure[idx].nodes = walk_graph(context.procedure[idx].nodes, fill_child_datacmds)
        context.procedure[idx].nodes = walk_graph(context.procedure[idx].nodes, calc_exprs)
                
                