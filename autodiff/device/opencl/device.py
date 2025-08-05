from .. import Device
from ...context import Proc
from ...alloc import AllocEntry, DeallocEntry
from ...graph import *
#from .kernels import *
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
                print("   ** SELECTED **")
                break

        assert self.device is not None, "Device none!"

        # initialize context and queue
        self.context = initialize_context(self.device)
        self.queue = create_command_queue(self.device, self.context)

        self.kernels = {}
        self.funcs = {} 
        self.buffers = {}

    def init (self, cmd):
        match cmd:
            case AllocEntry():
                if not cmd.is_temp:
                    self.buffers[cmd.id] = init_buffer(self.context, cmd.size, cmd.content)
            case DeallocEntry():
                pass
            case ForNode(): # handled by upper level
                pass

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
        
        buf_one = init_buffer(self.context, 4, np.array([1,2,3,4], dtype=np.float32))
        buf_two = init_buffer(self.context, 4, np.array([1,2,3,4], dtype=np.float32))
        buf_three = init_buffer(self.context, 4, None)

        kernel, f = build_kernel(self, "test", """
        __kernel void test (
            __global float* a,
            __global float* b,
            __global float* c,
            __local float* v,
            int val
        ) {{
            const size_t _global_id = get_global_id(0);
            c[_global_id] = a[_global_id] + val * b[_global_id];
        }}
        """, [
            Buffer(buf_one),
            Buffer(buf_two),
            Buffer(buf_three),
            LocalMem(4),
            Int(3)
        ], (4,), None)

        f()

        waitAll(self.queue)

        a = read_buffer(self.queue, buf_three, 4)
        free_buffer(buf_one)
        free_buffer(buf_two)
        free_buffer(buf_three)

        free_kernel(kernel)

        print(a)


    def __del__ (self):
        # free buffers
        for buf in self.buffers.values():
            free_buffer(buf)
        print(f"Free {len(self.buffers)} buffers")

        # free kernels
        for kernel in self.kernels.values():
            free_kernel(kernel) 

        # Free everything else
        free_queue(self.queue)
        free_context(self.context)
        free_device(self.device)


