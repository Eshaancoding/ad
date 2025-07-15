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
        if prod(n.shape) == 0: continue
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
    from .graph.kernelize import kernalize_node
    # lock procedure
    context.lock_proc = True

    # apply graph-level optimizations
    context.apply_per_graph(opt_node)
    
    # prepare graph into kernel level optimizations
    # kernel fusion happens here as well
    context.apply_per_graph(kernalize_node)
    
    # Linearize
    
    # apply linear optimizations
    # Dep opt, mem opt, as well as some memory accessing regrouping if needed
    # see if you can make fusion better here as well (test)
    
    # Send procedure to device to be executed

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