from .tensor import *
from .graph.data.concat import ConcatNode
from .context import context
from math import prod
from typing import Callable

##########################################
## Autodiff operations
def concat (nodes: List[Tensor], dim: int) -> Tensor:
    # filter nodes by shape if none
    l = []    
    for n in nodes:
        if prod(n.shape()) == 0: continue
        l.append(n)     
        
    # else, concat
    n = l[0]
    for i in range(1, len(l)):
        n = ConcatNode(n, l[i], dim)

    return n

def dot (left: Node, right: Node) -> Node:
    from .graph.compute.dotprod import DotProdNode
    return DotProdNode(left, right)

def execute ():
    from .graph.opt import opt_node
    # lock procedure
    context.lock_proc = True

    # apply graph-level optimizations
    context.apply_per_node(opt_node)
    
    # convert to kernel matrix 
    
    # 

    pass

##########################################
## Control flow 
def ir_for (r: range, f: Callable):
    from .graph.control.ir_for import ForNode

    context.add_proc() 
    f()
    proc = context.pop_proc()
    
    # Initialize for loop. This will automatically be added within the context
    ForNode(proc, r) 


##########################################
## Misc
def proc ():
    return context.procedure[0]