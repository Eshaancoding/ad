from . import nvrtc, cuda

from dataclasses import dataclass
import ctypes
import numpy as np
from typing import List, Optional

"""
A lot of this code is from tinygrad... thanks!
"""

use_latest = True # depends on driver verson. Good idea to check automatically or something

def check(status):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {ctypes.string_at(init_c_var(ctypes.POINTER(ctypes.c_char)(), lambda x: cuda.cuGetErrorString(status, ctypes.byref(x)))).decode()}")  # noqa: E501

def init_c_var(ctypes_var, creat_cb): return (creat_cb(ctypes_var), ctypes_var)[1]

def _get_bytes(arg, get_str, get_sz, check) -> bytes:
  sz = init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x))))
  return ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value)

def nvrtc_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(
        ctx, 
        nvrtc.nvrtcGetProgramLog, 
        nvrtc.nvrtcGetProgramLogSize, 
        lambda _: None
    ).decode() if ctx else ""
    raise Exception(f"Nvrtc Error {status}, {ctypes.string_at(nvrtc.nvrtcGetErrorString(status)).decode()}\n{err_log}")

def to_char_p_p(options: list[bytes], to_type=ctypes.c_char):
  return (ctypes.POINTER(to_type) * len(options))(*[ctypes.cast(ctypes.create_string_buffer(o), ctypes.POINTER(to_type)) for o in options])

####################### Initialization #######################
def init_device ():
    check(cuda.cuInit(0))
    return init_c_var(
        cuda.CUdevice(), 
        lambda x: check(cuda.cuDeviceGet(ctypes.byref(x), 0)) # only the first device for now
    )

def init_context (dev):
    f = cuda.cuCtxCreate_v2 if use_latest else cuda.cuCtxCreate
    
    ctx = init_c_var(
        cuda.CUcontext(), 
        lambda x: check(f(ctypes.byref(x), 0, dev))
    )
    check(cuda.cuCtxSetCurrent(ctx)) # set current ctx
    return ctx


def print_info (dev):
    # Get compute capability
    check(cuda.cuDeviceComputeCapability(
        ctypes.byref(major := ctypes.c_int()), 
        ctypes.byref(minor := ctypes.c_int()), 
        0 # only the first device for now
    ))
    
    # Get name of device <-- not sure about this code
    name = ctypes.string_at(init_c_var(
        ctypes.POINTER(ctypes.c_char)(), 
        lambda x: cuda.cuDeviceGetName(x, 100, dev)
    )).decode()

    arch = f"sm_{major.value}_{minor.value}"
    print(f"Using Cuda Device: {name} with architecture: {arch}")

    return arch

def free_context (context):
    f = cuda.cuCtxDestroy_v2 if use_latest else cuda.cuCtxDestroy
    f(context) 

####################### Memory #######################
# assumes that we are using f32; nothing else
def init_buffer (fp_size): 
    return init_c_var(
        cuda.CUdeviceptr(), 
        lambda x: check(cuda.cuMemAlloc_v2(ctypes.byref(x), fp_size*4))
    )

def read_buffers (buf_ptrs, shapes, offsets, func):
    """
    Note! Synchronous operation!
    """
    # not sure about this
    f = cuda.cuMemcpyDtoH_v2 if use_latest else cuda.cuMemcpyDtoH

    func_inps = []

    for idx in range(len(buf_ptrs)): 
        shape = shapes[idx]
        buf_ptr = buf_ptrs[idx]
        offset = offsets[idx]

        if offset > 0:
            # note: assumes F32!
            buf_ptr = cuda.CUdeviceptr(buf_ptr.value + offset*4)

        np_array = np.empty(shape, dtype=np.float32)
        dst_ptr = np_array.ctypes.data_as(ctypes.c_void_p)
        size_bytes = np_array.nbytes
        check(f(dst_ptr, buf_ptr, size_bytes, None))
        
        func_inps.append(np_array)

    func(*func_inps)

def write_buffer (buf_ptr, np_array:np.array, offset=0):
    # future: experiment with memcpy2d or memcpy3d. 
    # if it is faster, then go for it

    if offset > 0:
        # note: assumes F32!
        buf_ptr = cuda.CUdeviceptr(buf_ptr.value + offset*4)

    f = cuda.cuMemcpyHtoDAsync_v2 if use_latest else cuda.cuMemcpyHtoDAsync
    if not np_array.flags['C_CONTIGUOUS'] or np_array.dtype != np.float32:
        np_array = np.ascontiguousarray(np_array, dtype=np.float32)
    ptr = np_array.ctypes.data_as(ctypes.c_void_p)
    size_bytes = np_array.nbytes
    check(f(buf_ptr, ptr, size_bytes, None))

def free_buffer (ptr):
    f = cuda.cuMemFree_v2 if use_latest else cuda.cuMemFree
    check(f(ptr))

####################### Kernels #######################
class KernelArgs:
    pass

@dataclass
class Buffer (KernelArgs):
    buf: ctypes.c_uint64 # this is internally what cudeviceptr represents

@dataclass
class Int (KernelArgs):
    val: int

def build_kernel (
    name: str,
    program_source: str,
    args: List[KernelArgs],
    gpu_arch: str,
    global_size: List[int] = (1,1,1),
    local_size: List[int] = (1,1,1),
):
    compile_options = [f'--gpu-architecture={gpu_arch}']
    compile_options += ["-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include"]

    # Create program
    nvrtc_check(nvrtc.nvrtcCreateProgram(
        ctypes.byref(prog := nvrtc.nvrtcProgram()), 
        program_source.encode(), 
        "<null>".encode(), 
        0, 
        None, 
        None
    ))

    # compile program
    nvrtc_check(nvrtc.nvrtcCompileProgram(
        prog, 
        len(compile_options), 
        to_char_p_p([o.encode() for o in compile_options])
    ), prog)

    # get ptx
    ptx_size = ctypes.c_uint64()
    nvrtc_check(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)))
    ptx_buffer = (ctypes.c_char * ptx_size.value)()
    nvrtc_check(nvrtc.nvrtcGetPTX(prog, ptx_buffer))
    nvrtc_check(nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))) # destroy program; we have ptx

    # to get ptx as python string
    # ptx = ctypes.string_at(ptx_buffer, ptx_size.value).decode('utf-8').rstrip('\0') 

    # Load as function from PTX!
    nvrtc_check(nvrtc.cuModuleLoad(
        ctypes.byref(module := nvrtc.CUmodule()), 
        ptx_buffer
    ))

    nvrtc_check(nvrtc.cuModuleGetFunction(
        ctypes.byref(func := nvrtc.CUfunction()), 
        module, 
        ctypes.create_string_buffer(name.encode('utf-8'))
    ))

    # prepare parameters
    kernel_params = (ctypes.c_void_p * len(args))()
    for i, arg in enumerate(args):
        if isinstance(arg, Buffer):
            kernel_params[i] = ctypes.cast(ctypes.pointer(arg.buf), ctypes.c_void_p)
        elif isinstance(arg, Int):
            c_val = ctypes.c_int(arg.val)
            kernel_params[i] = ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p)
        
    # enqueue function
    return module, lambda: check(cuda.cuLaunchKernel(
        func, 
        *global_size,   # global size; x,y,z
        *local_size,    # local size; x,y,z
        0,              # dynamic shared mem bytes; not configured (
        None,           # stream to launch kernel; by default 0
        kernel_params,  # kernel args
        None            # extra args; we already supplied kernel_params
    ))

def free_module (mod):
    check(cuda.cuModuleUnload(mod))

def waitAll ():
    check(cuda.cuCtxSynchronize()) # hopefully this works lol
