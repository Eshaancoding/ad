from autodiff.device.cuda.cuda_helper import *
from .. import Device
from ...context import Proc
from ...graph import *
from ...fusion import *
from ...alloc import AllocEntry, DeallocEntry

class CudaDevice (Device):
    def __init__(self):
        super().__init__()
        self.modules = {}
        self.funcs = {}
        self.buffers = {}

        # get device
        dev = init_device()
        self.context = init_context(dev) 
        self.arch = print_info(dev)

    def init (self, cmd):
        from .kernels import init_dotprod, init_unary, init_binary, init_reduce, init_contigious, init_elwfuse, init_dp_elw_fuse, init_reduce_elw_fuse
        
        mod = None
        func = None

        match cmd:
            # both feeder and receiver should be done at the beginning
            case Feeder():
                # calculate the offset of the res
                cmd.kres.calc_offset()
                return
            case Receiver():
                # calculate the offset for each receiver; preperare for async 
                cmd.buffers = []
                cmd.shapes = []
                cmd.offsets = []
                for idx in range(len(cmd.kargs)):
                    cmd.kargs[idx].calc_offset()
                    karg = cmd.kargs[idx]
                    cmd.buffers.append(self.buffers[karg.id])
                    cmd.shapes.append(karg.shape)
                    cmd.offsets.append(karg.offset)
                return
            case AllocEntry():
                if not cmd.is_temp:
                    self.buffers[cmd.id] = init_buffer(self.context, cmd.size, cmd.content)
                return
            case DotProdNode(): mod, func = init_dotprod(self, cmd)
            case UnaryNode(): mod, func = init_unary(self, cmd)
            case BinaryNode(): mod, func = init_binary(self, cmd)
            case ReduceNode(): mod, func = init_reduce(self, cmd)
            case ContigiousNode(): mod, func = init_contigious(self, cmd)
            case ElwFuse(): mod, func = init_elwfuse(self, cmd)
            case DPElwFuse(): mod, func = init_dp_elw_fuse(self, cmd)
            case ReduceElwFuse(): mod, func = init_reduce_elw_fuse(self, cmd)
            case DeallocEntry(): return
            case ForNode(): return # handled by upper level

        assert (mod is not None and func is not None), f"Encountered unexpected cmd: {cmd}"

        self.modules[cmd.program_id] = mod
        self.funcs[cmd.program_id] = func

    def run (self, cmd):
        match cmd:
            case AllocEntry(): pass 
            case DeallocEntry(): pass
            case ForNode(): pass 
            case Feeder():
                arr = cmd.func()
                assert isinstance(arr, np.ndarray) and \
                        list(arr.shape) == list(cmd.shape), "Invalid function"

                write_buffer(
                    self.buffers[cmd.kres.id], 
                    arr, 
                    cmd.kres.offset
                )
            case Receiver():
                read_buffers(
                    cmd.buffers,
                    cmd.shapes,
                    cmd.offsets,
                    cmd.func
                )
            case _:
                self.funcs[cmd.program_id]() # enqueue to buffer

    def _run_proc (self, proc: Proc, func: Callable, init:bool=False):
        for cmd in proc.procedure:
            if isinstance(cmd, ForNode):
                assert (inner_proc := cmd.get_proc()) is not None, "Inner proc is None!"
                if init:
                    self._run_proc(inner_proc, func, init)
                else:
                    for _ in cmd.r:
                        self._run_proc(inner_proc, func, init)
            else:
                func(cmd)

    def execute (self, proc: Proc):
        self._run_proc(proc, self.init, init=True)
        self._run_proc(proc, self.run, init=False)
        waitAll()

    def __del__ (self):
        # free buffers
        for buf in self.buffers:
            free_buffer(buf)
        print(f"Freed {len(self.buffers)} buffers")

        # free modules (kernels in this case)
        for mod in self.modules.values():
            free_module(mod)
        print(f"Freed {len(self.modules)} kernels")

        # free everything else
        if hasattr(self.context):
            free_context(self.context)
