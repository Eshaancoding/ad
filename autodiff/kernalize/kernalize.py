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
               
def make_karg (initial_dim, child: Node, data_cmds, shape):
    if isinstance(child, ConstantNode):
        return KConstant(child.constant)
    elif isinstance(child, ConcatNode):
        dim = ndim_change_datacmds(initial_dim, data_cmds)
        idx_end = child.children_shapes[0][child.dim]

        condition = simplify_expr(MoreThan(dim[child.dim], Val(Constant(idx_end-1))))

        dim_two = deepcopy(dim)
        dim_two[child.dim] = Minus(dim_two[child.dim], Val(Constant(idx_end)))
        
        print("dim:", dim)
        print("dim_two:", dim_two)
        print()
        
        return KConcat(
            karg_one=make_karg(
                dim,
                child.left,
                child.children_datacmds[0],
                child.children_shapes[0]
            ),
            karg_two=make_karg(
                dim_two,
                child.right,
                child.children_datacmds[1],
                child.children_shapes[1]
            ),
            condition=condition
        )
        
    else:
        if initial_dim == Global():
            initial_dim = global_to_ndim(initial_dim, child.shape)
        dim = ndim_change_datacmds(initial_dim, data_cmds)
        return KMatrix(child.id, dim, shape)

def calc_exprs (node: Node, _): 
    match node:
        case DotProdNode() as n:
            # calc left
            n.kargs[0] = make_karg(
                initial_dim=[X(), Y()],
                child=n.left,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            )

            n.kargs[1] = make_karg(
                initial_dim=[X(), Y()],
                child=n.right,
                data_cmds=n.children_datacmds[1],
                shape=n.children_shapes[1]
            )

            # remove data cmds
            n.children_datacmds = None

        case BinaryNode() as n:
            # calc left
            n.kargs[0] = make_karg(
                initial_dim=Global(),
                child=n.left,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            )
            
            n.kargs[1] = make_karg(
                initial_dim=Global(),
                child=n.right,
                data_cmds=n.children_datacmds[1],
                shape=n.children_shapes[1]
            )

            # remove data cmds
            n.children_datacmds = None

        case ReduceNode() as n:
            # calc child
            n.kargs[0] = make_karg(
                initial_dim=[X(), Y()],
                child=n.child,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            )
            
            # remove data cmds
            n.children_datacmds = None
        
        case UnaryNode() | ContigiousNode() as n:
            # calc child
            n.kargs[0] = make_karg(
                initial_dim=Global(),
                child=n.child,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            )

            # remove data cmds
            n.children_datacmds = None
        
def kernalize (context: Context):
    for idx in range(len(context.procedure)):
        context.procedure[idx].nodes = walk_graph(context.procedure[idx].nodes, fill_child_datacmds)
        context.procedure[idx].nodes = walk_graph(context.procedure[idx].nodes, calc_exprs)
                
                