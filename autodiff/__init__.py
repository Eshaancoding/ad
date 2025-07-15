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
    from .graph.fuse import fuse_node
    from .graph.intermediate import opt_intermediate

    # lock procedure
    context.lock_proc = True

    # apply graph-level optimizations
    context.apply_per_node(opt_node)
    
    # Kernalize the graph; remove the data cmds and just use access expressions
    # at the point, all the binary ops can be represented by actual kernels
    context.apply_per_node(kernalize_node)
   
    # Find the repeated intermediate operations between graphs within the same block
    # extremely helpful for linearize. 
    # should be done before kernel fusion
    context.apply_per_block(opt_intermediate)
    
    # Kernel fusion
    context.apply_per_node(fuse_node)
    
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