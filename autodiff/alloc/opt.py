from . import *
from ..fusion import FuseBase

def temp_opt (proc: Proc):
    """
    Temp opt attempts to simplify the allocations and dealloaction. Example:
    
    ```
    Alloc a 128
    ... a computation ...
    Dealloc a 128
    Alloc b 128
    ... b computation ...
    Dealloc b 128
    ```
    
    this can simply be combined into:

    ```
    Alloc a 128
    ... a computation ...
    ... b computation, but replace b with a ...
    Dealloc b 128
    ```
    
    This optimization does not only apply to general allocations, but temp allocations as well
    """
    
    available: List[List[DeallocEntry]] = [[]]
    matches = {}

    def step_node (node: Node | AllocCmds): 
        if isinstance(node, AllocEntry):
            best_ty = None
            for avail in available[-1]:
                if  node.is_temp == avail.is_temp and \
                    avail.size == node.size and \
                    (avail.id, node.id) not in matches.items() and \
                    node.content is None:

                    best_ty = avail
                    break

            if best_ty is not None and best_ty.id not in matches:
                matches[best_ty.id] = node.id
                

        elif isinstance(node, DeallocEntry):
            available[-1].append(node)

    # iterate over the procedure, and fill in temp_reserved, temp_available, etc. as we go on
    # we had to create our own custom proc
    def step_proc (proc: Proc):
        for n in proc.procedure:
            if isinstance(n, FuseBase):
                for node in n.nodes:
                    step_node(node)                    
            elif isinstance(n, Node):
                if (p := n.get_proc()):
                    available.append([])
                    step_proc(p) 
                    available.pop()
                else:
                    step_node(n)
    
    step_proc(proc)

    # change for matrix
    def change_id (node: Node):
        if isinstance(node, Node):
            for to, fr in matches.items():
                node.rename(fr, to)
        return node
    proc.walk(change_id) 
    
    # remove the allocs and deallocs
    def remove_allocs_deallocs (node: Node):
        if isinstance(node, AllocEntry):
            if node.id in matches.values():
                return None

        # replace dealloc id with new
        if isinstance(node, DeallocEntry):
            for k, v in matches.items():
                if v == node.id:
                    node.id = k
                    break
                if k == node.id:
                    return None
    
        return node
    proc.walk(remove_allocs_deallocs) 

    return len(matches)