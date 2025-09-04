from typing import Callable, List, Optional
from ..expr import *
from colored import stylize, fore
from math import prod

class KernelArg:
    def __init__(self):
        pass
    
    def __repr__ (self):
        raise NotImplementedError("Kernel arg __repr__ not implemented")

    def is_none (self) -> bool:
        return False
    
    def get_ids (self, filter_temp=False) -> List[int]:
        raise NotImplementedError("Kernel arg get_ids not implemented")

    # helper functions across the opt (after linearize) that changes kernel arg
    def rename (self, fr, to, change_access: Optional[Callable[[Expression], Expression]] = None):
        raise NotImplementedError("rename not impl for base class")
    
    def change_to_temp (self, search:int):
        raise NotImplementedError("change_to_temp not impl for KernelArg")

    def calc_offset (self):
        raise NotImplementedError("calc_offset not available to kernel arg")

class NoneKernelArg (KernelArg):
    def __init__(self):
        super().__init__()
        
    def __repr__(self):
        return "None"

    def rename (self, fr, to, change_access: Optional[Callable[[Expression], Expression]] = None):
        raise NotImplementedError("Rename on none kernel arg")
    
    def is_none (self):
        return True

class KMatrix (KernelArg):
    def __init__(self, kern_id:int, access: Expression, shape: List[int]):
        self.id = kern_id
        self.access = access
        self.shape = shape
        self.is_temp = False
        self.offset = None # only filled out at calc_offset

    def __repr__ (self):
        if self.is_temp:
            t = stylize("Temp", fore("cyan"))
            return f"{t} {self.id}"
        else:
            return f"Mat ({stylize(f"id: {self.id}", fore("blue"))}, {self.access})"
    
    def get_ids (self, filter_temp=False):
        if filter_temp and self.is_temp:
            return []
        else:
            return [self.id]
    
    def rename (self, fr, to, change_access: Optional[Callable[[Expression], Expression]] = None):
        if self.id == fr:
            self.id = to
            if change_access is not None:
                self.access = change_access(self.access)
            
    def change_to_temp(self, search):
        if self.id == search:
            self.is_temp = True

    def calc_offset (self):
        match self.access:
            case Global():
                self.offset = 0
            case Add(Global(), Val(Constant(_ as c))):
                self.offset = c
            case Val(Constant(c)) if prod(self.shape) == 1:
                self.offset = c 
            case _:
                raise Exception(f"Invalid Feeder setting: {self.access}. Make sure to call .contigious before Global() ")

class KConcat (KernelArg):
    def __init__(self, karg_one: KernelArg, karg_two: KernelArg, condition: Expression, shape: List[int]):
        super().__init__()
        self.karg_one = karg_one
        self.karg_two = karg_two
        self.condition = condition # if True access karg_two; else false
        self.shape = shape

    def __repr__(self):
        from ..helper import indent_str
        
        # return f"Concat (\n{indent_str(self.karg_one)}\n{indent_str(self.karg_two)}\ncondition: {self.condition})"
        return f"\nConcat (\n{indent_str(f"1: {self.karg_one}\n2: {self.karg_two}\ncond: {self.condition}")}\n)\n"
    
    def get_ids (self, filter_temp=False):
        a = self.karg_one.get_ids(filter_temp)
        a.extend(self.karg_two.get_ids(filter_temp))
        return list(set(a)) 
    
    def rename (self, fr, to, change_access: Optional[Callable[[Expression], Expression]] = None):
        self.karg_one.rename(fr, to, change_access)
        self.karg_two.rename(fr, to, change_access)
        
    def change_to_temp(self, search):
        self.karg_one.change_to_temp(search)
        self.karg_two.change_to_temp(search)
    
class KConstant (KernelArg):
    def __init__(self, constant:float):
        super().__init__()
        self.constant = constant

    def __repr__ (self):
        return stylize(f"{self.constant}", fore("cadet_blue_2"))

    def get_ids (self, filter_temp=False):
        return []
    
    def rename (self, fr, to, change_access: Optional[Callable[[Expression], Expression]] = None):
        pass
    
    def change_to_temp(self, search):
        pass
