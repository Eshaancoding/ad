from copy import deepcopy
from typing import Callable, List, Dict, Union
from .node import Node
from .expr import *
from .graph import *
from time import time

# indent string
def indent_str(obj: any, size: int = 1, s: str = "    ") -> str:
    string = str(obj)
    indent = s * size
    lines = string.splitlines()
    indented_lines = [indent + line for line in lines]
    return '\n'.join(indented_lines)

# shape helper
def calc_stride(shape: List[int]) -> List[Expression]:
    n = len(shape)
    strides: List[Expression] = [Val(Constant(1)) for _ in range(n)]
    for i in reversed(range(n - 1)):
        next_val = strides[i + 1].get_const() * shape[i + 1]
        strides[i] = Val(Constant(next_val))
    return strides

def global_to_ndim(index: Expression, shape: List[int]) -> List[Expression]:
    strides = calc_stride(shape)
    return [
        Remainder(
            Div(index, strides[i]),
            Val(Constant(shape[i]))
        )
        for i in range(len(shape))
    ]

def ndim_to_global(dim: List[Expression], shape: List[int]) -> Expression:
    # assert len(dim) == len(shape), f"Dimension and the shape length mismatch\ndim: {dim}, shape: {shape}"

    strides = calc_stride(shape)
    global_expr = Mult(dim[0], strides[0])
    for i in range(1, len(shape)):
        global_expr = Add(
            global_expr,
            Mult(
                dim[i], 
                strides[i]
            )
        )
    return global_expr

def ndim_change_datacmds (dim: List[Expression], data_cmds: List[Node]):
    for data_cmd in data_cmds:
        match data_cmd:
            case PermuteNode(_, _ as permute_to):
                new_dim = [Val(Constant(0)) for _ in range(len(dim))]
                for i in range(len(dim)):
                    new_dim[i] = deepcopy(dim[permute_to[i]])
                dim = new_dim 
            
            case ViewNode(_, _ as target_dim) as n:
                dim = global_to_ndim(ndim_to_global(dim, target_dim), n.children_shapes[0])
                
            case BroadcastNode(_, _ as d, _):
                dim[d] = Val(Constant(0))
                
            case IndexNode(_, _ as start, _, _ as d):
                if start < 0: # in support for offsetted dim on concat
                    dim[d] = Minus(dim[d], Val(Constant(abs(start))))
                else:
                    dim[d] = Add(dim[d], Val(Constant(start)))
                
    return dim

def _walk_node (n: Node, visited: Dict[int, int], f: Callable, walk_block=True, walk_child=True, **kwargs) -> Node: 
    res = f(n, visited, **kwargs)    
    visited[n.id] = 1

    if res is None:
        return None
        
    if walk_block and (block := n.get_block()) is not None:
        n.block.nodes = walk_graph(block.nodes, f, walk_block, walk_child, **kwargs)
        n.block.nodes = list(filter(lambda x: x is not None, n.block.nodes))
        return n 

    if walk_child:
        if hasattr(res, "child"):
            if not (res.child.id in visited):
                res.child = _walk_node(res.child, visited, f, walk_block, walk_child, **kwargs)
        elif hasattr(res, "left") and hasattr(res, "right"):
            if not (res.left.id in visited):
                res.left = _walk_node(res.left, visited, f, walk_block, walk_child, **kwargs)
            if not (res.right.id in visited):
                res.right = _walk_node(res.right, visited, f, walk_block, walk_child, **kwargs)

    return res

def walk_graph (n: Node|List[Node]|Block, f: Callable, walk_block=True, walk_child=True, **kwargs) -> Union[List[Node], Node, Block]:
    """
    Walks through the graph of node or list of nodes. Includes options to walk over block (ex: for node) or child.
    Furthermore, the function could replace a node (just return a new one), or if returns None, removes the node from the list
    NOTE: If you are calling func within func at walk_graph, then make sure you have visited gaurd and insert node.id at visited
    NOTE: This uses .left, .right and .child. It will walk through .children() (ex: the Concat nodes, even though folded at internally kernalize), not the kargs_child_ids()
    """
    
    if isinstance(n, Block):
        n.nodes = walk_graph(n.nodes, f, walk_block, walk_child, **kwargs)
        return n
    elif isinstance(n, list) and len(n) > 0 and isinstance(n[0], Node):
        new_list = []
        visited = {}
        for node in n:
            node = _walk_node(node, visited, f, walk_block, walk_child, **kwargs)
            new_list.append(node)
        new_list = list(filter(lambda x: x is not None, new_list))
        return new_list
    elif isinstance(n, Node):
        return _walk_node(n, {}, f, walk_block, **kwargs)
    else:
        raise TypeError(f"Invalid type {type(n)} in format graph")

def benchmark (f: Callable, name: str):
    start = time() 
    r = f()
    end = time()
    print(f"{name} took: {(end - start)*1000:.4f} ms")
    return r
    
