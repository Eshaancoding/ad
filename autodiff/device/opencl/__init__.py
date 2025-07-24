from .. import Device
import pyopencl as cl
from .context import *
from ...context import Proc
from ...node import Node
from ...alloc import AllocEntry, DeallocEntry
from ...fusion import FuseBase

class OpenCLDevice (Device):
    def __init__(self, device: cl.device_type):
        super().__init__()
        self.context = ADCLContext(device) 
        
    def execute (self, proc: Proc):
        # do a warmup first (alloc, generate programs, etc.)
        self.context.dealloc_all()
                            
def execute_cmd (context: ADCLContext, cmd: any):
    if isinstance(cmd, AllocEntry):
        if not cmd.is_temp:
            context.alloc(cmd.id, cmd.size, cmd.content)
    elif isinstance(cmd, DeallocEntry):
        if not cmd.is_temp:
            context.dealloc(cmd.id)
    elif isinstance(cmd, Node):
        if cmd.get_block() is None:
            # execute specific node
            # for/if statements must be handled by top level (might need to do transfer from one device to another)
            pass
    elif isinstance(cmd, FuseBase):
        # execute specific fusion base
        pass  
    else:
        raise TypeError(f"Invalid type: {type(cmd)}")