from ..node import Node
from ..context import Context
from typing import List
from ..graph import *
from ..fusion import FuseBase

from ..fusion.helper import get_deps, get_res
from colored import stylize, fore, style

@dataclass 
class LocationBase:
    fused_id: Optional[int] # if in fused node
    control_node: Optional[int]
    
    def __hash__(self):
        return hash((self.fused_id, self.control_node))

@dataclass
class Location:
    loc: int
    fused_id: Optional[int] # if in fused node
    control_node: Optional[int]

    def __add__ (self, other):
        if type(other) == int:
            return Location(self.loc+other, fused_id=self.fused_id, control_node=self.control_node)
        else:
            raise TypeError(f"Invalid type for add loc: {type(other)}") 
        
    def base (self) -> LocationBase:
        return LocationBase(fused_id=self.fused_id, control_node=self.control_node)

class AllocEntry:
    def __init__(self, alloc_id: int, size: int, content: any = None):
        self.id = alloc_id
        self.size = size
        self.content = content
        self.is_temp = False
        
    def __repr__ (self):
        st = stylize(f"{"TEMP " if self.is_temp else ""}Alloc", fore("green")) + f" {self.id} " + stylize(str(self.size), fore("yellow") + style("bold"))
        if self.content is not None:
            st += " (has content)"    
        return st
  
class DeallocEntry:
    def __init__(self, dealloc_id: int, size: int):
        self.id = dealloc_id
        self.size = size
        self.is_temp = False

    def __repr__ (self):
        return stylize(f"{"TEMP " if self.is_temp else ""}Dealloc", fore("red")) + f" {self.id} " + stylize(str(self.size), fore("yellow") + style("bold"))
    

def alloc (proc: List[Node | FuseBase]):
    from .insert_alloc import insert_alloc
    from .temp_alloc import temp_alloc
    from .temp_opt import temp_opt

    # first, insert all the allocations
    fused_ids_to_f = insert_alloc(proc) # TODO: you have to support for and other control
    
    # Just continue to apply for every proc recursively
    temp_alloc(proc, fused_ids_to_f)
    
    temp_opt(proc)
    # temp_clean(proc)