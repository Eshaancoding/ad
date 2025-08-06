"""
Light helper functions that provides a wraper over the opencl clib
Definitely inspired by tinygrad opencl initialization + opencl.py. Thanks :D
"""

import ctypes
from typing import Optional, List
from . import opencl as cl
from enum import Enum
import numpy as np
from dataclasses import dataclass

############################# Initialization ############################# 

class CLDevice (Enum):
    GPU=0
    CPU=1
    ALL=2

cl_errors = {attr: k for k in dir(cl) if k.startswith("CL_") and isinstance(attr:=getattr(cl, k), int) and attr <= 0}
def check (status:int):
    if status != 0:
        raise RuntimeError(f"OpenCL Status != 0. Error {status}: {cl_errors.get(status, "Unknown error")}")

def to_size_t_arr (arr):
    SizeTArray = ctypes.c_size_t * len(arr)
    c_array = SizeTArray(*arr)
    return ctypes.cast(c_array, ctypes.POINTER(ctypes.c_size_t))

def get_platforms (): 
    # get platforms
    check(cl.clGetPlatformIDs(
        0, 
        None, 
        num_platforms := ctypes.c_uint32()
    ))

    check(cl.clGetPlatformIDs(
        num_platforms.value, 
        platform_ids := (cl.cl_platform_id * num_platforms.value)(), 
        None
    ))

    return platform_ids

def get_devices (platform: cl.struct__cl_platform_id):
    check(cl.clGetDeviceIDs(
        platform, 
        cl.CL_DEVICE_TYPE_ALL, 
        0, 
        None, 
        num_devices := ctypes.c_uint32()
    ))
    
    check(cl.clGetDeviceIDs(
        platform,
        cl.CL_DEVICE_TYPE_ALL, 
        num_devices, 
        devices := (cl.cl_device_id * num_devices.value)(),
        None
    ))

    return devices

def get_queue (platform):
    # get num devices
    check(cl.clGetDeviceIDs(
        platform,
        cl.CL_DEVICE_TYPE_ALL, 
        0, 
        None, 
        num_devices := ctypes.c_uint32()
    ))

    check(cl.clGetDeviceIDs(
        platform,
        cl.CL_DEVICE_TYPE_ALL, 
        num_devices, 
        devices := (cl.cl_device_id * num_devices.value)(), 
        None
    )) 

    return devices

def get_device_name (device: cl.struct__cl_context):
    check(cl.clGetDeviceInfo(
        device, 
        cl.CL_DEVICE_NAME, 
        256, 
        buf := ctypes.create_string_buffer(256), 
        None
    ))

    return buf.value.decode()

def get_device_type (device: cl.struct__cl_context) -> CLDevice:
    check(cl.clGetDeviceInfo(
        device, 
        cl.CL_DEVICE_TYPE, 
        ctypes.sizeof(cl.cl_device_type()), 
        buf := ctypes.pointer(cl.cl_device_type()), 
        None
    ))

    match buf.contents.value: 
        case cl.CL_DEVICE_TYPE_CPU:
            return CLDevice.CPU
        case cl.CL_DEVICE_TYPE_GPU:
            return CLDevice.GPU
        case _:
            raise RuntimeError("Invalid device type!")

def initialize_context (device: cl.struct__cl_device_id):
    context = cl.clCreateContext(
        None, 
        1, 
        device,
        cl.clCreateContext.argtypes[3](), 
        None, 
        status := ctypes.c_int32()
    )
    check(status.value)
    return context

def create_command_queue (device: cl.struct__cl_device_id, context: cl.struct__cl_context):
    # add conditinal profiling later?
    queue = cl.clCreateCommandQueue(
        context, 
        device, 
        #cl.CL_QUEUE_PROFILING_ENABLE,
        0,
        status := ctypes.c_int32()
    )

    check(status.value)
    return queue

def free_queue (queue: cl.struct__cl_command_queue):
    cl.clReleaseCommandQueue(queue)

def free_context (context: cl.struct__cl_context):
    cl.clReleaseContext(context)

def free_device (device: cl.struct__cl_device_id):
    cl.clReleaseDevice(device)

############################ Memory ############################# 
def init_buffer (context: cl.struct__cl_context, size: int, content: Optional[np.array]):
    """
    Creates buffer object using opencl. Note that this assumes fp32 array   
    """
    
    if content is not None:
        assert content.dtype == np.float32, "Content must be a float32 array!"
        float_ptr = content.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        void_ptr = ctypes.cast(float_ptr, ctypes.c_void_p)

    buf = cl.clCreateBuffer(
        context,
        cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR if content is not None else cl.CL_MEM_READ_WRITE,
        size * 4, # 4 is the size of float32 (default precision). Needs to be changed for different fp types
        void_ptr if content is not None else None,
        err_code := ctypes.pointer(cl.cl_int())
    )

    check(err_code.contents.value)
    return buf

def free_buffer (buffer: cl.struct__cl_mem):
    check(cl.clReleaseMemObject(buffer))

def read_buffer (
    command_queue: cl.struct__cl_command_queue, 
    buffer: cl.struct__cl_mem, 
    size: int,
    shape: Optional[List[int]] = None
):
    """
    Reads buffer object using opencl. Note that this is a BLOCKING operation.
    """

    out_array = np.empty(shape if shape is not None else (size,), dtype=np.float32)
    out_ptr = out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    size = ctypes.c_size_t(out_array.nbytes)

    check(cl.clEnqueueReadBuffer(
        command_queue,      # command queue pointer
        buffer,             # buffer pointer
        True,               # blocking_read = CL_TRUE
        0,                  # offset
        size,               # size
        out_ptr,            # pointer to host memory
        0,                  # num_events_in_wait_list
        None,               # event_wait_list
        None                # event
    ))

    return out_array

############################ Kernels ############################# 
"""
clCreateProgramWithSource
clBuildProgram
clSetKernelArg
clEnqueueNDRangeKernel
"""

class KernelArgs:
    pass

@dataclass
class Buffer (KernelArgs):
    buf: cl.struct__cl_mem

@dataclass
class LocalMem (KernelArgs):
    size: int

@dataclass
class Int (KernelArgs):
    val: int

def build_kernel (
    dev,
    name: str,
    program_source: str,
    args: List[KernelArgs],
    global_size: List[int],
    local_size: Optional[List[int]]
):
    # First, create the program from source
    program = cl.clCreateProgramWithSource(
        dev.context,
        1,
        ctypes.cast(
            ctypes.create_string_buffer(program_source.encode("utf-8")),
            ctypes.POINTER(ctypes.c_char)
        ),
        None,
        errcode := ctypes.pointer(cl.cl_int())
    )

    check(errcode.contents.value)
  
    # Second, build program into kernel
    status = cl.clBuildProgram(
        program,
        1,
        dev.device,
        None, # no options as of now
        cl.clBuildProgram.argtypes[4](),
        None
    )

    # Third, get the kernel
    kernel = cl.clCreateKernel(
        program,
        ctypes.cast(
            ctypes.create_string_buffer(name.encode("utf-8")),
            ctypes.POINTER(ctypes.c_char)
        ),
        errcode
    )

    check(cl.clReleaseProgram(program)); # release program for memory benefits
    check(errcode.contents.value)

    # Nice status message for error-handling build program (copied from tinygrad)
    if status != 0:
        cl.clGetProgramBuildInfo(program, dev.device, cl.CL_PROGRAM_BUILD_LOG, 0, None, log_size := ctypes.c_size_t())
        cl.clGetProgramBuildInfo(program, dev.device, cl.CL_PROGRAM_BUILD_LOG, log_size.value, mstr := ctypes.create_string_buffer(log_size.value), None) 
        raise RuntimeError(f"OpenCL Compile Error\n\n{mstr.value.decode()}")

    # Set kernel arguments to the program
    for idx, arg in enumerate(args):
        obj = None
        arg_size = None # in bytes
        match arg:
            case Buffer(_ as buf_obj):
                obj = ctypes.byref(buf_obj)
                arg_size = ctypes.sizeof(buf_obj)
            case LocalMem(_ as size):
                obj = None
                arg_size = size * 4 # again, assuming fp32
            case Int(_ as val):
                obj = ctypes.pointer(cl.cl_int(val))
                arg_size = ctypes.sizeof(cl.cl_int())
            case _:
                raise Exception(f"Invalid argument {arg} at idx: {idx}")

        try:
            check(cl.clSetKernelArg(
                kernel,
                idx,
                arg_size,
                obj
            ))
        except Exception as e:
            print("****** INVALID ARG *****")
            print(f"Idx: {idx}, arg: {arg}")
            print(f"Object: {obj}")
            print(f"Arg size: {arg_size}")
            raise e

    if global_size is not None and local_size is not None:
        assert len(global_size) == len(local_size), "dimensions of global size and local size must be equal"

    assert len(global_size) > 0, "Must specify a global size"

    # Enqueue kernel (as a function)
    return kernel, lambda: check(cl.clEnqueueNDRangeKernel(
        dev.queue,
        kernel,
        len(global_size),
        None,
        to_size_t_arr(global_size),
        None if local_size is None else to_size_t_arr(local_size),
        0,
        None,
        None
    ))

def waitAll (command_queue: cl.struct__cl_command_queue):
    cl.clFinish(command_queue)

def free_kernel (kernel: cl.struct__cl_kernel):
    cl.clReleaseKernel(kernel)
