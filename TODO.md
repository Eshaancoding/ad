# TODO

## Main Todo

* Make sure dep node gets replaced if there's any changes to the node

* numerical test for neural networks in general 
    * do it for at least neural net
    * make sure it is consistent as well
    * Although, shows promise already

* Better fusion?
    * Fusion is pretty weird, not going to lie. 
        * toposort gaur at the last step - converting to procedure
        * many to one / one to many resolve  
    * could be better improved...

* Then do device feeder
    * see if you can train a LLM faster than normal
    * async transfer?
    * Something similar to tinyllm etc.

====== Then test with more tests (use pytest) ====== 
* add more control
* control, if, etc. etc. etc.

## Backend
    
* **Kernel**:
    * Divergent branching conflicts
        * Could be more complicated on various scenarious (sources, vars, concat, trackers, etc.)        
        * you have to know what is right and what is not right to be fair.

    * **MEM OPTS**: Swizzling memory
        * dependending on access patterns IF there's like a single type of access pattern
            * if multiple, you might need to just rely on those different access patterns
            * just check and whether you can optimize

        * Remove movement opt if.. 
            * alr know contigious when 1. same id 2. same access expression

    * **OPTS**: Swapping accessing expressions between input and output of two adjacent kernels?
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

    * **Kernel experimentation:**
        * Experiment with different parameters of dot prod + other kernels 
            * [this](https://mesozoic-egg.github.io/tinygrad-notes/20241203_beam.html) does a good job
            * there's other optimizations, I am sure. Don't focus on that right now, have the general base for everything first.

        * Contigious memory vs. direct accessing for dot prod kernels
            * efficient dot product kernels assume that it is contigious 
            * Furthermore, we assume that that `A` in `Ax` in matrix multiplication is **column-wise** rather than **row-wise**
                * need to manually assume that there's a transpose before the A in matrix multiplication.

        * How/where to organize this? Each device will have different kernels which will have different params to opts...
            * probably within each device?  
            * **YOU NEED BOTH**

    * **Kernel Fusion**
        * ~~do basic multiple binary/unary kernel fusion~~
        * dot prod impl is kinda wacky
            * access expression assumes global id...
        * anyway to do dotprod fusion?  
            * similar to flash attention
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
            * life is not all that simple now is it hehe
        * you need more knowledge of all of this before you go into this optimizations
            * not sure if you can beat hand-tune optimizations
        * There's recent work on kernel fusion of **everything** somehow
            * however, you'd have to handle your own GPU synchronization. **THIS CAN BE A BENEFIT**.

 * **Node Opts**
    * Concat + view operations can be streamlined
        * this is mostly due to **concat**. If I am being honest, there's probably a better way for implementing backward pass for concat? I believe
            * maybe not 
        * Should be replaced in graphical format
        * main problem with transformer implementation right now (and well, any other implementations)

    * View removal
        * If multiple views in sequence, just turn it into the one single view (the last view operation)
        * if view is already in shape, then delete

    * Remove double permutations
        * if `b = a.permute([1, 0])` followed by `c = b.permute([1, 0])`, this is the same as the original input `a`
        * transpose via transpose

    * More aggressive IR optimizations for localization:
        * Also test for RNN, Transformers --> improve library
        * Then you can improve the IR optimizations as well. 
            * etc. etc. etc. 

    * Prox opt doesn't have safegaurd for IF/While? (tests still work surprisingly)


* **General Ideas**: 
    * Dynamic Shape

    * Cuda graphs
        * have to use BR compiler hints in order to determine while or if statement
        * you may also need to send extra compiler hints 
        * skdjfksjdfkjsdkfjskdfjksdjf that's also going to be pretty weird.

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
