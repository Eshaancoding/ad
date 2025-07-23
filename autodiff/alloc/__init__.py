from ..context import Proc
from ..graph import *
from colored import stylize, fore, style

@dataclass 
class LocationBase:
    fused_id: Optional[int] # if in fused node
    proc_id: Optional[int]  # if in nested procedure (for, if, etc.)
    
    def __hash__(self):
        return hash((self.fused_id, self.proc_id))

@dataclass
class Location:
    loc: int
    fused_id: Optional[int] # if in fused node
    proc_id: Optional[int]

    def __add__ (self, other):
        if type(other) == int:
            return Location(self.loc+other, fused_id=self.fused_id, proc_id=self.proc_id)
        else:
            raise TypeError(f"Invalid type for add loc: {type(other)}") 
        
    def base (self) -> LocationBase:
        return LocationBase(fused_id=self.fused_id, proc_id=self.proc_id)

class AllocCmds:
    pass

class AllocEntry (AllocCmds):
    def __init__(self, alloc_id: int, size: int, content: any = None):
        super().__init__()
        self.id = alloc_id
        self.size = size
        self.content = content
        self.is_temp = False
        
    def __repr__ (self):
        st = stylize(f"{"TEMP " if self.is_temp else ""}Alloc", fore("green")) + f" {self.id} " + stylize(str(self.size), fore("yellow") + style("bold"))
        if self.content is not None:
            st += " (has content)"    
        return st
  
class DeallocEntry (AllocCmds):
    def __init__(self, dealloc_id: int, size: int):
        super().__init__()
        self.id = dealloc_id
        self.size = size
        self.is_temp = False

    def __repr__ (self):
        return stylize(f"{"TEMP " if self.is_temp else ""}Dealloc", fore("red")) + f" {self.id} " + stylize(str(self.size), fore("yellow") + style("bold"))
    
def alloc (proc: Proc):
    from .insert import insert_alloc
    from .temp import temp_alloc
    from .opt import temp_opt
    from .clean import temp_clean

    # first, insert all the allocations
    fused_ids_to_f = insert_alloc(proc) 
    
    # Just continue to apply for every proc recursively
    temp_alloc(proc, fused_ids_to_f)
    
    count = 1
    while count > 0: 
        # temp opt could be better... 
        count = temp_opt(proc)
    temp_clean(proc)