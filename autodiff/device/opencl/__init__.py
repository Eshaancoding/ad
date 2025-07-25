from .. import Device
import pyopencl as cl
from .context import *
from ...context import Proc
from ...node import Node
from ...alloc import AllocEntry, DeallocEntry
from ...fusion import FuseBase
from ...graph import *
from .kernels import *
from time import time

class OpenCLDevice (Device):
    def __init__(self, device: cl.device_type):
        super().__init__()
        self.context = ADCLContext(device) 
        
    def _exec_proc (self, proc: Proc, warm_up:bool=False):
        for cmd in proc.procedure:
            if isinstance(cmd, ForNode):
                assert (inner_proc := cmd.get_proc()) is not None, "Inner proc is None!"
                if warm_up:
                    self._exec_proc(inner_proc)
                else:
                    for _ in cmd.r:
                        self._exec_proc(inner_proc)
            else:
                execute_cmd(self.context, cmd)
        
    def execute (self, proc: Proc):
        # warmup
        print("Warmup...")
        self._exec_proc(proc, warm_up=True) 

        print("Executing...")
        start = time()
        self._exec_proc(proc)     
    
        # finish     
        self.context.finish() # todo: experiment whether you can enqueue copy from the dep list (put this cmd after...)
        print("elapsed: ", time()-start)
            
        # dealloc 
        self.context.dealloc_all()

def execute_cmd (context: ADCLContext, cmd):
    match cmd:
        case AllocEntry():
            if not cmd.is_temp:
                context.alloc(cmd.id, cmd.size, cmd.content)
        case DeallocEntry():
            pass
        case ForNode(): # handled by upper level
            pass
        case BinaryNode():
            execute_binary(context, cmd)                     
        case UnaryNode():
            execute_unary(context, cmd)
        case ContigiousNode():
            execute_contigious(context, cmd)
        case DotProdNode():
            execute_dotprod(context, cmd)
        case ReduceNode():
            execute_reduce(context, cmd)
        case ElwFuse():
            execute_elwfuse(context, cmd)
        case DPElwFuse():
            execute_dp_elw_fuse(context, cmd)
        case ReduceElwFuse():
            execute_reduce_elw_fuse(context, cmd)
        case _:
            raise Exception(f"Invalid cmd: {cmd}")