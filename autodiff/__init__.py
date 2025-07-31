from autodiff.fusion.base import FuseBase
from autodiff.opt import dep_opt, mem_opt, repeat_opt
from autodiff.opt.simplify import simpl_node
from .context import context
from math import prod
from typing import Callable
from .graph import Tensor, ConcatNode, Node
from typing import List
from .helper import benchmark, walk_graph

##########################################
## Autodiff operations
def concat (nodes: List[Tensor], dim: int) -> Tensor|ConcatNode:
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
    # from .core.kernelize import kernalize
    from .kernalize.kernalize import kernalize
    from .linearize import linearize
    from .alloc import alloc
    from .device.opencl import OpenCLDevice, cl

    from pprint import pprint

    # lock procedure
    context.lock_proc = True

    # perform optimizations 
    benchmark(lambda: dep_opt(context), "dep_opt")    # delete nodes that are not needed or computed
    benchmark(lambda: repeat_opt(context), "repeat_opt") # re-use nodes already computed

    # apply graph-level optimizations (ex: constant simplification)
    benchmark(lambda: simpl_node(context), "simplify node")

    # Kernalize the graph; remove the data cmds and just use access expressions
    # From this point on, each children node should rely on kwargs_child_id rather than iterating over children (because of Concat)
    # in future releases, we can have the capabiltiy for nodes to have more than 3 childrens. However, for now this is not implemented
    benchmark(lambda: kernalize(context), "kernalize")

    # Linearize + fusion
    proc = benchmark(lambda: linearize(context.main_proc()), "linearize")

    # perform memory optimization
    proc = benchmark(lambda: mem_opt(proc), "mem opt")

    # apply linear optimizations
    # Dep opt, mem opt, as well as some memory accessing regrouping if needed
    # see if you can make fusion better here as well (test)
    # Apply allocations + opts on allocs
    alloc(proc)

    # assign program id for each node that is about to be executed
    def assign_program_id (n: Node, _):
        if isinstance(n, Node) or isinstance(n, FuseBase):
            n.program_id = context.get_prog_id()
        return n
    proc.walk(assign_program_id, step_fused=False, step_proc=True)

    pprint(proc)
    
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

def set_lenient_dep ():
    """
    lenient_dep will add gradients to dependencies.
    If set to False, even if you call .backward(),
    it won't calculate any gradients unless it is explicitly used in another computation.
    (ex: performing gradient update with the original tensor)
    """
    context.lenient_dep = True
