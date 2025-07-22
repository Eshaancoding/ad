from . import *

def temp_opt (proc: List[Node | FuseBase]):
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
    This operation is applied per-procedural wise
    """
    
    # First, get the lifetimes of all allocs and deallocs
    
    
    # Then, attempt to combine the lifetimes until we can't anymore 
    
    # Apply lifetimes (delete allocs + deallocs) 