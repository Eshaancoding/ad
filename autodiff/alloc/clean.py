from . import *

def temp_clean (proc: Proc):
    # record all the allocs and deallocs. Then insert at the beginning
    allocs = [] 
    deallocs = []
    
    def inner (node: Node):
        if isinstance(node, AllocEntry):
            if not node.is_temp:
                allocs.append(node) 
            else:
                return node
        elif isinstance(node, DeallocEntry):
            if not node.is_temp:
                deallocs.append(node)
        else:
            return node

        return None
    
    proc.walk(inner)
    
    for alloc in allocs:
        proc.insert(0, alloc)
        
    for dealloc in deallocs:
        proc.append(dealloc)