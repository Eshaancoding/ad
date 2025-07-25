from .context import context
from math import prod
from typing import Callable
from .graph import Tensor, ConcatNode, Node
from typing import List

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
    from .graph import DotProdNode
    return DotProdNode(left, right)

def execute ():
    from .opt import opt_node
    # from .core.kernelize import kernalize
    from .kernalize.kernalize import kernalize
    from .linearize import linearize
    from .alloc import alloc
    from .device.opencl import OpenCLDevice, cl

    from pprint import pprint

    # lock procedure
    context.lock_proc = True
    
    # apply graph-level optimizations'
    # context.apply_per_node(opt_node)
    
    # Kernalize the graph; remove the data cmds and just use access expressions
    # From this point on, each children node should rely on kwargs_child_id rather than iterating over children (because of Concat)
    # in future releases, we can have the capabiltiy for nodes to have more than 3 childrens. However, for now this is not implemented 
    kernalize(context)
    
    # Linearize + fusion
    proc = linearize(context.main_proc()) 
    
    # apply linear optimizations
    # Dep opt, mem opt, as well as some memory accessing regrouping if needed
    # see if you can make fusion better here as well (test)
    
    # Apply allocations + opts on allocs
    alloc(proc)
    
    # pprint(proc)
    
    # Send procedure to device to be executed
    OpenCLDevice(cl.device_type.ALL).execute(proc)

##########################################
## Control flow 
def ir_for (r: range, f: Callable):
    from .graph import ForNode
    context.add_proc() 
    f()
    proc = context.pop_proc()
    
    # Initialize for loop. This will automatically be added within the context
    ForNode(proc, r) 


##########################################
## Misc
def print_graph ():
    """
    Note that print graph will show concat node, even though it's folded at kernalize.
    """
    context.print_graph()