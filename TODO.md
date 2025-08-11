# TODO

## Main Todo

* Feeder (host --> device)
    * call via function
    * per variable basis --> create a new tensor entirely (will be replaced anyways) 

* Receiver (device --> host)
    * call via function
    * can replace dep opts? technically

* Indexing via a non constant node
    * can't do slices --> requires dynamic shapes
    * add pytest for this

* index available at "for" node 
    * store copies at CPU + device.

----- performance testing? save to drive  -----

* Numerical tests 
    * different opts + neural networks, etc.

    * To fix errors:
        * on linearize, support += nodes (id being reused)
        * make sure you calculate deps accordingly (ex: alloc)
        * etc. etc. etc. Make sure you do it accordingly

* Better fusion?
    * Fusion is pretty weird, not going to lie. 
        * toposort gaur at the last step - converting to procedure
        * many to one / one to many resolve  
    * attempt to make it as EASY AS POSSBILE to implement operator fusion
    * could be better improved...

* add more control
    * match (selection w/ different expressons)
    * while (w/ expressions)

    * Divergent branching conflicts
        * if different, push the computation match/if statement itself

    * **DONE** for loops (known at compile time)
    * **DONE**  Functions + function call

* Then do async transfer
    * see if you can train a LLM faster than normal

* **MAJOR**: Kernel experimentation
    * look into what tinygrad has done and generalize that (pretty nice generalization)
    * contigious vs. not contigious memory (faster, or not?)
    * ideally, try to make it as easy as possible for kernel experimentation (like kernel fusion)

* **MAJOR**: dynamic shapes
    * somehow find a way such that you can still run tetris opt... 
        * each "tetris opt instance" must have an unique set of dynamic memory shape that can be optimized
        * ex: (N,5) and (N,3) can be optimized together, but not (N,N)
    * furthermore, you can override what future instructions are in the first place
        * but...then the chip needs a feeder? interesting
        * Current idea for chip: increase the instruction word size such that you can repeat instructions w/ different bases
            * There will be multiple registers dedicated towards # of repeats or if single repeat
            * this way, only requires like 1 extra clock cycle, which is really nice
            * this could also be applied by 
    * Different backends handle different shapes accordingly
        * on opencl, might need to perform CPU + GPU transfer
        * on tensorrt by nvidia, there's specific guidelines for this as well (different execution profiles without CPU intervention?)
        * more research needs to be done in that area
    * then, you can add dynamic slices with indexing, etc.

---------------------- done before stanford? ----------------------

* Tensor/multi-gpu sharding

* Add CUDA support + advanced dotprod (probably need an external device lowk)
    * this is where you are going more into the backend kernel space
    * look more into kernel experimentation (look below) etc.

* Chip project
    * Study the computations needed around popular algos
        * See if you can improve the design of the chip with that (ex: accessing memory for non-contigious tensors) 
    * Read more about chip-to-chip interconnect  
        * Per-layer sharding --> top-down approach
            * this is good for training especially
        * across-layer sharding --> side-by-side approach
            * ex: splitting heads

## Extra Links:

**Better kernel fusion**:
* look into optimized [link](https://siboehm.com/articles/22/CUDA-MMM)
* even better optimization for [kernels](https://salykova.github.io/sgemm-gpu)
* technically, there's even more [kernels at llm.c](https://github.com/karpathy/llm.c/tree/master/dev/cuda)
* more kernel opt (+ read kernel fusion) [here](https://mesozoic-egg.github.io/tinygrad-notes/20241203_beam.html)
* transpose operator faster: [here](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/)
    * prolly uses this: [here](https://veitner.bearblog.dev/tma-introduction/)
* even faster kernel stuff for generation: [here](https://www.together.ai/blog/chipmunk)
    * The entire purpose of **TogetherAI** is optimizing kernels in a way.
* There are more and more special features of hardware on more and more GPUs:
    * [link](https://tridao.me/blog/2024/flash3/)

**Kernel experimentation**

* [this](https://mesozoic-egg.github.io/tinygrad-notes/20241203_beam.html) does a good job

## Experiments
    
* **MEM OPTS**: Swizzling memory
    * dependending on access patterns IF there's like a single type of access pattern
        * if multiple, you might need to just rely on those different access patterns
        * just check and whether you can optimize
        * if so, move to the if statement itself. 

    * Remove movement opt if.. 
        * alr know contigious when 1. same id 2. same access expression

* Swapping accessing expressions between input and output of two adjacent kernels?
    * is it even possible? 

* Memory experiments needed (do this movement/no movement experiment after kernel fusion)
    * **You should test whether a weird write is slower than a fast write + movement**
        * There's specialize transpose kernels as well...
        * you could try experimenting with that.
        * is there cases where fast write + movement is better?

    * **ALSO TEST**
        * which of the following procedures is best
            * dot product (uncontigious write) --> sum --> dot product (uncontigious read)
            * dot product (uncontigious write) --> sum --> movement --> dot product (contigious read)
            * dot product (contigious write) --> movement --> sum --> movement --> dot product (contigious read)

            * etc. etc. etc. 

## Frontend

* ~~Softmax~~
* ~~ReLU~~
* Different losses: 
    * Cross entropy
    * MSE
    * etc.
* ~~RmsNorm   <-- Needed for Transformer~~
* ~~LayerNorm <-- Needed for Transformer~~
* Dropout   <-- Dropout
* All of the above operations should be enough for building up to attention
* Transformer
* Implement Adam
    * apparently there's a general impl of optimizers that impls Adam and others...
        * check that!
* batch matrix multiplication
    * ~~view with -1~~
* NN operations:
    * Convolutions
    * ConvTranspose
    * Max/Avg/Fractional Pooling layers <-- create general pooling operator (like op)
    * zero padding --> concat under the hood
    * Activations:
        * ELU
        * GeLU
        * SeLU
    * Batch Norm
    * GNN/Cell,RNN/Cell,etc.
    * More Loss funcs:
        * L1Loss
        * NLLLoss
        * KLDiv loss
    * .product(); like .sum()?
* Other operations?
    * vstack, hstack <-- simple wrapper over cat
    * split tensor
    * tile
* DataLoader (use feeder)
* Simple operations:
    * abs
    * random trig/almost trig funcs: 
        * logs
        * acosh
        * acos
        * inverse tan
        * etc. <-- do this when you know how to function with weird funcs
    * argmax/argmin
    * min/max <-- simple wrapper over idx
    * more random distribution generators
        * berournelli, etc.
