from . import *

def temp_clean (proc: Proc):
    # record all the allocs and deallocs. Then insert at the beginning
    allocs = [] 
    deallocs = []
    
    def inner (node: Node, _):
        if isinstance(node, AllocEntry):
            if node.is_temp or node.content is not None:
                return node
            else:            
                allocs.append(node) 
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
