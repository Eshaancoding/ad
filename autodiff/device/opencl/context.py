import pyopencl as cl
from typing import Dict
from typing import List
import numpy as np

# AutoDiff OpenCL Context
class ADCLContext ():
    def __init__(self, device_type: cl.device_type):
        self.ctx = cl.Context(dev_type=device_type)
        device = self.ctx.devices[0]
        # print("Using device:", device.name)
        # print("  Vendor:", device.vendor)
        # print("  Type:", cl.device_type.to_string(device.type))
        
        self.command_queue = cl.CommandQueue(context=self.ctx)
        self.buffers: Dict[int, cl.Buffer] = {}
        self.programs: Dict[str, cl.Program] = {}
        self.programs_str: Dict[str, str] = {}
        self.size_dict: Dict[int, int] = {}
        self.program_id = -1
        
    def get_prog_id (self) -> int:
        self.program_id += 1
        return self.program_id
        
    def alloc (self, buf_id: int, size: int, content:any=None) -> cl.Buffer:
        if buf_id not in self.buffers:
            bf = cl.Buffer(
                self.ctx, 
                cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR if content is not None else cl.mem_flags.READ_WRITE, 
                size=size * np.dtype(np.float32).itemsize,
                hostbuf=np.array(content, dtype=np.float32) if content is not None else None
            )
            self.buffers[buf_id] = bf
            self.size_dict[buf_id] = size
            return bf
        elif content is not None:
            if len(content) != self.size_dict[buf_id]:
                self.dealloc_all()
                raise Exception("Size mismatch at alloc")
            cl.enqueue_copy(self.command_queue, self.buffers[buf_id], content)

        # else, it has already allocated the buffer
        return self.buffers[buf_id]

    def get_buffer (self, buf_id:int) -> cl.Buffer:
        if buf_id not in self.buffers:
            self.dealloc_all()
            raise Exception("Content buffer id is not in context")       
        return self.buffers[buf_id]
    
    def get_contents (self, buf_id:int) -> np.array:
        """NOTE: waits for event!"""
        if buf_id not in self.buffers:
            self.dealloc_all()
            raise Exception(f"Content buffer id {buf_id} is not in context (len: {len(self.buffers)})")

        res_g = np.empty(shape=[self.size_dict[buf_id], ], dtype=np.float32)
        cl.enqueue_copy(self.command_queue, res_g, self.buffers[buf_id])
        return res_g
    
    def get_program (self, name:str, program:str) -> cl.Program:
        if name not in self.programs: 
            kernels = cl.Program(self.ctx, program).build().all_kernels()
            assert len(kernels) == 1, "More than one kernel defined"
            self.programs[name] = kernels[0]
            self.programs_str[name] = program

        return self.programs[name] 
        
    def finish (self):
        self.command_queue.finish()
        
    # dealloc 
    def dealloc (self, buf_id):
        if buf_id not in self.buffers:
            self.dealloc_all()
            raise Exception("Content buffer id is not in context")
        self.buffers[buf_id].release()
        del self.buffers[buf_id]

    def dealloc_all (self): 
        for buf in self.buffers.values():
            buf.release()
        self.buffers.clear()
        
    def __del__ (self):
        self.dealloc_all()
