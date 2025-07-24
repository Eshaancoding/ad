import pyopencl as cl
from typing import Dict
from typing import List

# AutoDiff OpenCL Context
class ADCLContext ():
    def __init__(self, device_type: cl.device_type):
        self.ctx = cl.Context(dev_type=device_type)
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
            bf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=size, hostbuf=content)
            self.buffers[buf_id] = bf
            self.size_dict[buf_id] = size
            return bf
        elif content is not None:
            if len(content) != self.size_dict[buf_id]:
                self.dealloc_all()
                raise Exception("Size mismatch at alloc")

            cl.enqueue_copy(self.ctx, self.buffers[buf_id], content)

        # else, it has already allocated the buffer
        return self.buffers[buf_id]

    def get_buffer (self, buf_id:int) -> cl.Buffer:
        if buf_id not in self.buffers:
            self.dealloc_all()
            raise Exception("Content buffer id is not in context")       
        return self.buffers[buf_id]
    
    def get_contents (self, buf_id:int) -> List[float]:
        if buf_id not in self.buffers:
            self.dealloc_all()
            raise Exception("Content buffer id is not in context")
        res_g = [0.0 for _ in range(self.size_dict[buf_id])]
        cl.enqueue_copy(self.ctx, res_g, self.buffers[buf_id])
        return res_g
    
    def get_program (self, name:str, program:str) -> cl.Program:
        if name not in self.programs: 
            self.programs[name] = cl.Program(self.ctx, program).build()
            self.programs_str[name] = program
        else:
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
