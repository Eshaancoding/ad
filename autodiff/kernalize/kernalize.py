from autodiff.graph.data.feeder import Feeder
from ..expr import *
from ..expr.simplify import simplify_expr
from ..helper import global_to_ndim, ndim_to_global, walk_graph, ndim_change_datacmds
from ..node import Node
from ..context import Context, context

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
        case PermuteNode() | ViewNode() | BroadcastNode() | IndexNode() as n:
            if n.id in context.deps:
                raise Exception("A non contigious node is added to dependency. Use .contigious()")
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

            return n
        case ReduceNode() | UnaryNode() | ContigiousNode() as n:
            # handle child node
            while True:
                n.child, cmd, did_repl = pot_child_datacmd(n.child)
                if isinstance(cmd, Node):
                    n.children_datacmds[0].append(cmd)
                if not did_repl:
                    break

            return n
    return node
               
def make_karg (initial_dim, child: Node, data_cmds, shape, size:Optional[int]=None):
    if isinstance(child, ConstantNode):
        return KConstant(child.constant)
    elif isinstance(child, ConcatNode):
        dim = ndim_change_datacmds(initial_dim, data_cmds)        
        concat_dim = child.dim
        idx_end = child.children_shapes[0][concat_dim]
        
        dim_two = deepcopy(dim)
        dim_two[concat_dim] = Minus(dim_two[concat_dim], Val(Constant(idx_end)))
   
        condition = deepcopy(MoreThan(dim[concat_dim], Val(Constant(idx_end-1))))
    
        karg_one = make_karg(
            dim,
            child.left,
            child.children_datacmds[0],
            child.children_shapes[0]
        )
        
        karg_two=make_karg(
            dim_two,
            child.right,
            child.children_datacmds[1],
            child.children_shapes[1]
        )
        
        return KConcat(
            karg_one,
            karg_two,
            condition=condition,
            shape=shape
        )
        
    else:
        dim = ndim_change_datacmds(initial_dim, data_cmds)
        dim = simplify_expr(ndim_to_global(dim, child.shape), size)
        return KMatrix(child.id, dim, child.shape)
   
def make_res_arg (kern_id:int, is_global:bool, shape:list) -> KernelArg:
    if is_global:
        return KMatrix(
            kern_id=kern_id,
            access=Global(),
            shape=shape
        )
    else:
        return KMatrix(
            kern_id=kern_id,
            access=simplify_expr(ndim_to_global([X(), Y()], shape), None),
            shape=shape
        )
   
def calc_exprs (node: Node, _):
    match node:
        case DotProdNode() as n:
            assert n.children_datacmds is not None

            # calc left
            n.kargs[0] = make_karg(
                initial_dim=[X(), Y()],
                child=n.left,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            )

            # calc right
            n.kargs[1] = make_karg(
                initial_dim=[X(), Y()],
                child=n.right,
                data_cmds=n.children_datacmds[1],
                shape=n.children_shapes[1]
            )
            
            # calculate res arg
            n.kres = make_res_arg(n.id, is_global=False, shape=n.shape)

            # remove data cmds
            n.children_datacmds = None

        case BinaryNode() as n:
            size = math.prod(n.shape) 
            assert n.children_datacmds is not None

            # calc left
            n.kargs[0] = make_karg(
                initial_dim=global_to_ndim(Global(), n.children_shapes[0]),
                child=n.left,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0],
                size=size
            )
            
            n.kargs[1] = make_karg(
                initial_dim=global_to_ndim(Global(), n.children_shapes[1]),
                child=n.right,
                data_cmds=n.children_datacmds[1],
                shape=n.children_shapes[1],
                size=size
            )
            
            # calculate res arg
            n.kres = make_res_arg(n.id, is_global=True, shape=n.shape)

            # remove data cmds
            n.children_datacmds = None

        case ReduceNode() as n:
            assert n.children_datacmds is not None

            # calc child
            n.kargs[0] = make_karg(
                initial_dim=[X(), Y()],
                child=n.child,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0]
            )
            
            # calculate res arg
            n.kres = make_res_arg(n.id, is_global=False, shape=n.shape)
            
            # remove data cmds
            n.children_datacmds = None
        
        case UnaryNode() | ContigiousNode() as n:
            size = math.prod(n.shape) 
            
            # calc child
            n.kargs[0] = make_karg(
                initial_dim=global_to_ndim(Global(), n.children_shapes[0]),
                child=n.child,
                data_cmds=n.children_datacmds[0],
                shape=n.children_shapes[0],
                size=size
            )
            
            # calculate res arg
            n.kres = make_res_arg(n.id, is_global=True, shape=n.shape)

            # remove data cmds
            n.children_datacmds = None

        # set kres for feeder
        case Feeder() as n:
            n.kres = make_res_arg(n.id, is_global=True, shape=n.shape) 

    return node
        
def kernalize (context: Context):
    for idx in range(len(context.procedure)):
        context.procedure[idx].nodes = walk_graph(context.procedure[idx].nodes, fill_child_datacmds)
        context.procedure[idx].nodes = walk_graph(context.procedure[idx].nodes, calc_exprs)
