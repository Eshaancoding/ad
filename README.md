# Ahead-of-Time Reverse Mode Automatic Differentation Compiler 

## Features:
1. Only dependency is `numpy` (handles Tensor storage)! 
2. Implemented basic control loops (for, if, etc.)
3. Reduces ML operations (LayerNorm, Transformer, etc.) into a few simple operations (Dot Prod, Sum, Add, Feeder, Unary, etc.)
    * This is declared in `autodiff/graph`.
4. Two intermediate representation:
   1. Graph IR (good for simplifying operations; dependency optimization)
   2. Procedural IR (actual execution; kernel fusion, memory optimizations) 
5. Transformers implementation! See `examples/transformer.py`
6. Basic Kernel Fusion (see `autodiff/fusion/ops.py`)
   1. Dot Prod + Elementwise operations
   2. Reduce + Elementwise Operations
   3. Multiple ElwFuse Operations
7. Optimizations (see `autodiff/opt/`)
   1. Dependency optimization (unused computation $\to$ prune)
   2. Simplify optimization (constant/unary folding)
   3. Memory Optimization (reuse same memory address if allows) $\to$ `autodiff/opt/mem_opt.py` and `autodiff/alloc/tetris_opt.py`
      1. Typically saves 70% of memory (including forward and backward pass)!
      2. Running `examples/transformer.py` yeilds `ALLOC DEBUG: Using: 11238 out of 36758
ALLOC DEBUG: Saved 25520. Using 30.573%`
   4. Repeat Optimizations (reuse same memory if already computation already computed)
   5. More to come!
8. Easy to add multiple backends $\to$ just need to declare a few operations
   1. However, for fast inference you would want to implement fused kernels.
   2. OpenCL backend implemented (working on CUDA)

Still long way to go! See `TODO.md`

## Some Preliminary Results
For short transformers, our implementation beats it! Comparing `examples/transformer.py` and `py_test/transformer_opt.py`:

1. 1-layer transformer
    * our lib: 1.32 seconds
    * pytorch compile 2.0: 3.58 seconds
2. 2-layer transformer
    * our lib: 2.71 seconds
    * pytorch compile 2.0: 4.22 seconds
3. 3-layer transformer
    * our lib: 4.17 seconds
    * pytorch compile 2.0: 4.82 seconds
4. 4-layer transformer
    * execution time for our lib: 5.195 seconds
    * pytorch compile 2.0: 5.68 seconds

* it makes sense that pytorch will eventually catch up
    * the matmul kernel themselves are not optimized! OpenCL Kernels I wrote are very naive, see `autodiff/device/opencl/kernels`.

## Example

Take `examples/linear.py`: 

```py
from autodiff import execute, ir_for, Feeder, Receiver, print_graph
import autodiff
from autodiff.nn import Linear, Sequential, Sigmoid, TransformerEncoder, SGD 
from autodiff.nn.activations.relu import ReLU
from autodiff.nn.transformer import * 
from autodiff.helper import benchmark
from autodiff.print_graph import pg
import numpy as np

autodiff.graph.tensor.is_testing = True

nn = Sequential(
    Linear(255, 255),
    Sigmoid(),
    Linear(255, 255)
)

def print_inp (res):
    pass

def save_params (*args):
    print(len(args))
    for arg in args:
        print(arg)

opt = SGD(nn.parameters(), lr=0.01)
def f():
    opt.zero_grad()
    val = Feeder(
        lambda: np.full((2,255), 0.2, dtype=np.float32), 
        shape=[2,255]
    )
    res = nn(val)
    res.backward() 
    opt.step()

benchmark(lambda: ir_for(range(0, 1000), f), name="Tracking nodes")
Receiver(save_params, opt.parameters, name="saving params")
benchmark(lambda: execute(), name="full exec")
```

`Feeder` is the node that enables CPU --> GPU transfer. Furthermore, `Receiver` is the node that enables GPU --> CPU transfer. By default, `Feeder` and `Receiver` is asynchronous! 

This gets converted into Graphical IR:

```
0: Tensor(id: 0, orig_shape: [255, 255])
1: Tensor(id: 1, orig_shape: [255])
2: Tensor(id: 2, orig_shape: [255, 255])
3: Tensor(id: 3, orig_shape: [255])
4: 74 = For 0 to 1000:
  0: 58 = ADD IN PLACE (65025) 
    Tensor(id: 0, orig_shape: [255, 255])
    77 = MULT (65025) 
      45 = Dot Prod [255, 2] x [2, 255] 
        44 = Permute [1, 0] 
          4 = Feeder "" (dim: [2, 255])
        79 = MULT (510) 
          37 = MULT (510) 
            36 = MULT (510) 
              34 = MULT (510) 
                25 = Dot Prod [2, 255] x [255, 255] 
                  23 = Const(val: 1.0, dim: [2, 255])
                  24 = Permute [1, 0] 
                    Tensor(id: 2, orig_shape: [255, 255])
                33 = MULT (510) 
                  31 = RECIP (510) 
                    30 = MULT (510) 
                      15 = ADD (510) 
                        13 = EXP2 (510) 
                          81 = MULT (510) 
                            8 = ADD (510) 
                              5 = Dot Prod [2, 255] x [255, 255] 
                                4 = Intermediate
                                0 = Intermediate
                              7 = Broadcast dim 0 to size 2 
                                6 = View from [255] to [1, 255] 
                                  Tensor(id: 1, orig_shape: [255])
                            80 = Const(val: -1.4426950408889634, dim: [2, 255])
                        14 = Const(val: 1.0, dim: [2, 255])
                      15 = Intermediate
                  32 = Const(val: -1.0, dim: [2, 255])
              35 = Const(val: 0.6931471805599453, dim: [2, 255])
            13 = Intermediate
          78 = Const(val: -1.4426950408889634, dim: [2, 255])
      76 = Const(val: -0.01, dim: [255, 255])
  1: 63 = ADD IN PLACE (255) 
    1 = Intermediate
    83 = MULT (255) 
      49 = View from [1, 255] to [255] 
        48 = View from [255] to [1, 255] 
          47 = SUM on dim: -1 (Vec/X: 255, Reduce/Y: 2) 
            46 = Permute [1, 0] 
              79 = Intermediate
      82 = Const(val: -0.01, dim: [255])
  2: 68 = ADD IN PLACE (65025) 
    2 = Intermediate
    87 = MULT (65025) 
      27 = Dot Prod [255, 2] x [2, 255] 
        26 = Permute [1, 0] 
          16 = RECIP (510) 
            15 = Intermediate
        23 = Intermediate
      86 = Const(val: -0.01, dim: [255, 255])
  3: 73 = ADD IN PLACE (255) 
    Tensor(id: 3, orig_shape: [255])
    89 = MULT (255) 
      53 = View from [1, 255] to [255] 
        52 = View from [255] to [1, 255] 
          51 = SUM on dim: -1 (Vec/X: 255, Reduce/Y: 2) 
            50 = Permute [1, 0] 
              23 = Intermediate
      88 = Const(val: -0.01, dim: [255])
5: 75 = Receiver "saving params" 
  58 = Intermediate
  63 = Intermediate
  68 = Intermediate
  73 = Intermediate
```

Which is then converted procedurally (with Kernel Fusion and access expressions):

```
Tensor(id: 0, orig_shape: [255, 255])
Tensor(id: 1, orig_shape: [255])
Tensor(id: 2, orig_shape: [255, 255])
Tensor(id: 3, orig_shape: [255])
74 = For 0 to 1000:
    Mat (id: 25, ((X * 255) + Y)) = Dot Prod [2, 255] x [255, 255] --> (1.0, Mat (id: 2, ((Y * 255) + X)))
    ReduceElwFuse Fusion (fuse_id: 98):
        Mat (id: 51, X) = SUM on dim: -1 (Vec/X: 255, Reduce/Y: 2) --> (1.0)
        Mat (id: 89, Global) = MULT (255) --> (Mat (id: 51, (((Global % 255) % 255) % 255)), -0.01)
        Mat (id: 3, Global) = ADD IN PLACE (255) --> (Mat (id: 3, Global), Mat (id: 89, Global))
    Mat (id: 4, Global) = Feeder "" (dim: [2, 255])
    DPElwFuse Fusion (fuse_id: 94):
        Mat (id: 5, ((X * 255) + Y)) = Dot Prod [2, 255] x [255, 255] --> (Mat (id: 4, ((X * 255) + Y)), Mat (id: 0, ((X * 255) + Y)))
        Mat (id: 8, Global) = ADD (510) --> (Mat (id: 5, Global), Mat (id: 1, ((Global % 255) % 255)))
        Mat (id: 81, Global) = MULT (510) --> (Mat (id: 8, Global), -1.4426950408889634)
        Mat (id: 13, Global) = EXP2 (510) --> (Mat (id: 81, Global))
        Mat (id: 15, Global) = ADD (510) --> (Mat (id: 13, Global), 1.0)
        Mat (id: 16, Global) = RECIP (510) --> (Mat (id: 15, Global))
        Mat (id: 30, Global) = MULT (510) --> (Mat (id: 15, Global), Mat (id: 15, Global))
        Mat (id: 31, Global) = RECIP (510) --> (Mat (id: 30, Global))
        Mat (id: 33, Global) = MULT (510) --> (Mat (id: 31, Global), -1.0)
        Mat (id: 34, Global) = MULT (510) --> (Mat (id: 25, Global), Mat (id: 33, Global))
        Mat (id: 36, Global) = MULT (510) --> (Mat (id: 34, Global), 0.6931471805599453)
        Mat (id: 37, Global) = MULT (510) --> (Mat (id: 36, Global), Mat (id: 13, Global))
        Mat (id: 79, Global) = MULT (510) --> (Mat (id: 37, Global), -1.4426950408889634)
    DPElwFuse Fusion (fuse_id: 96):
        Mat (id: 45, ((X * 255) + Y)) = Dot Prod [255, 2] x [2, 255] --> (Mat (id: 4, ((Y * 255) + X)), Mat (id: 79, ((X * 255) + Y)))
        Mat (id: 77, Global) = MULT (65025) --> (Mat (id: 45, Global), -0.01)
        Mat (id: 0, Global) = ADD IN PLACE (65025) --> (Mat (id: 0, Global), Mat (id: 77, Global))
    ReduceElwFuse Fusion (fuse_id: 99):
        Mat (id: 47, X) = SUM on dim: -1 (Vec/X: 255, Reduce/Y: 2) --> (Mat (id: 79, ((Y * 255) + X)))
        Mat (id: 83, Global) = MULT (255) --> (Mat (id: 47, (((Global % 255) % 255) % 255)), -0.01)
        Mat (id: 1, Global) = ADD IN PLACE (255) --> (Mat (id: 1, Global), Mat (id: 83, Global))
    DPElwFuse Fusion (fuse_id: 95):
        Mat (id: 27, ((X * 255) + Y)) = Dot Prod [255, 2] x [2, 255] --> (Mat (id: 16, ((Y * 255) + X)), 1.0)
        Mat (id: 87, Global) = MULT (65025) --> (Mat (id: 27, Global), -0.01)
        Mat (id: 2, Global) = ADD IN PLACE (65025) --> (Mat (id: 2, Global), Mat (id: 87, Global))
75 = Receiver "saving params" --> (Mat (id: 0, Global), Mat (id: 1, Global), Mat (id: 2, Global), Mat (id: 3, Global))
```

After determining allocation (code on `autodiff/alloc`), we execute on device! 