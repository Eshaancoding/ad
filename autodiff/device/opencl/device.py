from .. import Device
from ...context import Proc, context
from ...alloc import AllocEntry, DeallocEntry
from ...graph import *
from ...fusion import *
from .cl_helper import *
from typing import Callable

class OpenCLDevice (Device):
    def __init__(self, sel: CLDevice):
        super().__init__()
        
        platforms = get_platforms()
        assert len(platforms) > 0, "No platforms available!"

        # get devices
        devices = get_devices(platforms[0])
        
        # For platform ids, get the name of device and initialize device
        print()
        self.device = None
        for device in devices:
            device_name = get_device_name(device)
            device_type = get_device_type(device)

            print(f"Device name: {device_name}")
            print(f"   Device type: {device_type}")

            if sel == CLDevice.ALL or sel == device_type:
                self.device = device
                print("   *** SELECTED ***")
                break

        assert self.device is not None, "Device none!"

        # initialize context and queue
        self.context = initialize_context(self.device)
        self.queue = create_command_queue(self.device, self.context)

        # other variables
        self.kernels = {}
        self.funcs = {} 
        self.buffers = {}
        self.buffers_size = {}

        # read buffers events
        self.read_buffer_events = []
        self.read_func_pointer = []
        self.read_outputs = []

    def init (self, cmd):
        from .kernels import init_dotprod, init_unary, init_binary, init_reduce, init_contigious, init_elwfuse, init_dp_elw_fuse, init_reduce_elw_fuse
        
        kern = None
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
                cmd.sizes = []
                cmd.shapes = []
                cmd.offsets = []
                for idx in range(len(cmd.kargs)):
                    cmd.kargs[idx].calc_offset()
                    karg = cmd.kargs[idx]
                    cmd.buffers.append(self.buffers[karg.id])
                    cmd.sizes.append(self.buffers_size[karg.id])
                    cmd.shapes.append(karg.shape)
                    cmd.offsets.append(karg.offset)
                return
            case AllocEntry():
                if not cmd.is_temp:
                    self.buffers[cmd.id] = init_buffer(self.context, cmd.size, cmd.content)
                    self.buffers_size[cmd.id] = cmd.size
                return
            case DotProdNode(): kern, func = init_dotprod(self, cmd)
            case UnaryNode(): kern, func = init_unary(self, cmd)
            case BinaryNode(): kern, func = init_binary(self, cmd)
            case ReduceNode(): kern, func = init_reduce(self, cmd)
            case ContigiousNode(): kern, func = init_contigious(self, cmd)
            case ElwFuse(): kern, func = init_elwfuse(self, cmd)
            case DPElwFuse(): kern, func = init_dp_elw_fuse(self, cmd)
            case ReduceElwFuse(): kern, func = init_reduce_elw_fuse(self, cmd)
            case DeallocEntry(): return
            case ForNode(): return # handled by upper level

        assert (kern is not None and func is not None), f"Encountered unexpected cmd: {cmd}"

        self.kernels[cmd.program_id] = kern
        self.funcs[cmd.program_id] = func

    def run (self, cmd):
        match cmd:
            case AllocEntry(): pass 
            case DeallocEntry(): pass
            case ForNode(): pass 
            case Feeder():
                arr = cmd.func()
                assert isinstance(arr, np.ndarray) and list(arr.shape) == list(cmd.shape), "Invalid function"
                write_buffer(self.queue, self.buffers[cmd.kres.id], cmd.kres.offset, arr)
            case Receiver():
                read_buffers(
                    self,
                    cmd.buffers,
                    cmd.sizes,
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
        
        waitAll(self.queue)
        #waitForEvents(self.read_buffer_events)

    def __del__ (self):
        # free buffers
        for buf in self.buffers.values():
            free_buffer(buf)
        print(f"Freed {len(self.buffers)} buffers")

        # free kernels
        for kernel in self.kernels.values():
            free_kernel(kernel) 
        print(f"Freed {len(self.kernels)} kernels")

        # Free everything else
        free_queue(self.queue)
        free_context(self.context)
        free_device(self.device)


