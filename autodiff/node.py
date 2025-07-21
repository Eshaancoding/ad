from copy import deepcopy
import math
from .expr import Expression, NoneExpr
from typing import List, Callable, Dict
from .kernalize import NoneKernelArg, KernelArg

class Node:
    ############################################################
    ## Derived methods/init
    # TODO: Add res expr and shape to super().__init__()
    def __init__(self, children, shape:List[int]):
        from autodiff import context

        self.id = context.get_id()
        self.children_exprs: List[Expression] = []
        self.children_shapes: List[List[int]] = []
        self.children_datacmds: List[List[Node]] = [] # will be only datacmds nodes
        self.kargs: List[KernelArg] = []
        for ch in children:
            if not isinstance(ch, Node): 
                raise TypeError("Children is not type of node!")
            
            # record child shapes + set up child_exprs 
            self.children_exprs.append(NoneExpr()) # will be filled out at kernalize
            
            # this shape could change (especially at kernalize). Save the deepcopy of shape and use this shape
            # One tensor can be represented in multiple dimensions/views (depends on the children exprs)
            self.children_shapes.append(deepcopy(ch.shape)) 
            
            # fill in children datacmds. This is an internal variable needed at kernalize
            self.children_datacmds.append([])
            
            # fill in kargs. Filled in at kernalize
            self.kargs.append(NoneKernelArg())

            # as we are creating nodes, we record the latest node being changed within the context
            # as we go through the computation core, the order of the nodes being changed will also be recorded
            # and being recorded into a procedure
            context.remove_from_dep(ch)
        context.add_to_dep(self)

        # set the self.left, self.right, or self.child at children
        if len(children) == 1:
            self.child = children[0]
        elif len(children) == 2:
            self.left = children[0]
            self.right = children[1]
        elif len(children) != 0:
            raise TypeError("Children length is invalid (expected one or two)")
        
        self.shape = shape
        
    def bck (self, grad):
        raise NotImplementedError
    
    def __repr__ (self) -> str:
        raise NotImplementedError
    
    ############################################################
    ## Comparisons
    def id_eq (self, other):
        assert isinstance(other, Node), "id eq input invalid"
        return self.id == other.id
    
    ############################################################
    ## Other methods
    # Calls backend 
    def backward (self):
        from .graph import ConstantNode
        self.bck(ConstantNode(1.0, self.shape))
    
    # helper for binary ops
    def _to_node (self, node, dim: List[int]):
        if isinstance(node, int):
            if dim is not None:
                from .graph import ConstantNode
                return ConstantNode(float(node), dim)
            else:
                raise TypeError("Cannot run dot product on a constant without dim (declare constant manually)")
        elif isinstance(node, float):
            if dim is not None:
                from .graph import ConstantNode
                return ConstantNode(node, dim)
            else:
                raise TypeError("Cannot run dot product on a constant without dim (declare constant manually)")
        elif not isinstance(node, Node):
            raise TypeError(f"Invalid type {type(node)} when converting to node {node}")
       
        return node 
    
    # helper to iterateover the children of nodes
    def walk (self, f: Callable, visited: Dict[int, int], args=[]):
        res = f(self, *args) 
        visited[self.id] = 1
       
        if hasattr(res, "child"):
            if not (res.child.id in visited):
                res.child = res.child.walk(f, visited=visited, args=args)
        elif hasattr(res, "left") and hasattr(res, "right"):
            if not (res.left.id in visited):
                res.left = res.left.walk(f, visited=visited, args=args)

            if not (res.right.id in visited):
                res.right = res.right.walk(f, visited=visited, args=args)
            
        return res
    
    # helper to get children 
    def children (self):
        if hasattr(self, "left") and hasattr(self, "right"):
            return [self.left, self.right]
        elif hasattr(self, "child"):
            return [self.child]
        else:
            return []

    ############################################################
    ## Binary operations (+, *, -, /, @ matmul)
    def __add__ (self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self, self._to_node(other, self.shape))
        return BinaryNode(a, b, BinaryOp.ADD) 
    
    def __radd__ (self, other):
        return self.__add__(other)
    
    def __mul__ (self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self, self._to_node(other, self.shape))
        return BinaryNode(a, b, BinaryOp.MULT)
    
    def __rmul__ (self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self, self._to_node(other, self.shape) * -1.0)
        return BinaryNode(a, b, BinaryOp.ADD)
    
    def __rsub__(self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self._to_node(other, self.shape), self * -1.0)
        return BinaryNode(a, b, BinaryOp.ADD)
    
    def __truediv__(self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self, self._to_node(other, self.shape).recip())
        return BinaryNode(a, b, BinaryOp.MULT)
    
    def __rtruediv__(self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self.recip(), self._to_node(other, self.shape))
        return BinaryNode(a, b, BinaryOp.MULT)
    
    def __neg__ (self):
        return self.__mul__(-1.0)
    
    def __matmul__ (self, other):
        from .graph import DotProdNode
        return DotProdNode(self, self._to_node(other, None))
    
    def __rmatmul__ (self, other):
        from .graph import DotProdNode
        return DotProdNode(self._to_node(other, None), self)
    
    ############################################################
    ## Comparison operations
    def __eq__(self, other):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.EQUAL)

    def __lt__(self, other):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.LESS_ZERO)

    def __le__(self, other):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.LESS_OR_EQ_ZERO)

    def __gt__(self, other):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.MORE_ZERO)

    def __ge__(self, other):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.MORE_OR_EQ_ZERO)
    
    ############################################################
    ## Unary operations
    def exp2 (self):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.EXP2)
    
    def log2 (self):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.LOG2)
    
    def sin (self):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.SIN)
    
    def sqrt (self):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.SQRT)
    
    def recip (self):
        from .graph import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.RECIP)
    
    ## Operations derived from the above core ops
    def cos (self):
        return ((math.pi/2) - self).sin()
    
    def exp (self):
        return (self * (1.0 / math.log(2.0))).exp2()
        
    def ln (self):
        return self.log2() * math.log(2.0)
        
    def pow2 (self): 
        return self * self
        
    def pow3 (self):
        return self * self.pow2()
        
    def powf (self, p: float):
        if p == 2.0:
            return self.pow2()
        elif p == 3.0:
            return self.pow3()
        else:
            return (p * self.log2()).exp2()
        
    ############################################################
    ## Data Manipulation operations
    def broadcast (self, dim: int, size: int):
        from .graph import BroadcastNode
        assert self.shape[dim] == 1, "Broadcast dim invalid"
        return BroadcastNode(self, dim, size)
    
    def view (self, target_dim: list[int]):
        from .graph import ViewNode
        # handle -1 dim 
        target_dim = ViewNode.handle_minus_dim(self.shape, target_dim)

        if self.shape != target_dim:
            return ViewNode(self, target_dim)
        else:
            return self
        
    def contigious (self):
        from .graph import ContigiousNode
        return ContigiousNode(self)
    
    def __getitem__(self, idx):
        from .graph import IndexNode
        # Ensure indexing is a tuple
        if not isinstance(idx, tuple):
            idx = (idx,)
        
        result = self
        dim_counter = 0  # Tracks which dimension we're on

        for i in idx:
            if isinstance(i, slice):
                start = 0 if i.start is None else i.start
                sh = self.shape[dim_counter]
                end = sh if i.stop is None else i.stop
                
                if sh != end - start:
                    result = IndexNode(result, start, end, dim_counter)
                dim_counter += 1
            elif isinstance(i, int):
                # Convert integer index to a slice [i:i+1]
                result = IndexNode(result, i, i+1, dim_counter)
                result = result.squeeze(dim_counter)
                # Do NOT increment dim_counter here: dimensions are reduced
                # For example, a[:, 2, :] -> dim=1 is squeezed out
            else:
                raise TypeError(f"Unsupported index type: {type(i)}")
        
        return result

    def permute (self, l: list[int]):
        from .graph import PermuteNode
        
        # if contigious list ([0,1,2,..])
        if sum([l[i] == i for i in range(len(l))]) == len(l):
            return self
        else:
            return PermuteNode(self, l)
    
    def _reduce_proc (self, dim: int, op):
        from .graph import ReduceNode
        p_dim = len(self.shape) 

        # Handle negative dimension
        if dim < 0:
            dim = p_dim + dim

        # ========= permute target dim to last =========
        # e.g. for dim = 2 and p_dim = 5
        # [0, 1, 2, 3, 4] --> [0, 1, 4, 3, 2]
        permute_to = list(range(p_dim))
        permute_to[dim], permute_to[-1] = permute_to[-1], permute_to[dim]

        node = self.permute(permute_to)

        # ========= view as 2D [other_dims, target_dim_size] =========
        new_dim = deepcopy(node.shape)
        last_dim = new_dim[-1]
        node = node.view([-1, last_dim])

        # ========= sum along last dimension =========
        node = ReduceNode(node, op)

        # ========= unview to remove last dim =========
        new_dim.pop()
        node = node.view(new_dim)

        # ========= unpermute =========
        # undo the earlier swap of dim <--> last
        permute_to_x = deepcopy(permute_to)
        del permute_to_x[dim]
        node = node.permute(permute_to_x)
        
        return node
    
    def sum (self, dim: int):
        from .graph import ReduceOp
        return Node._reduce_proc(self, dim, ReduceOp.SUM)
   
    def max (self, dim: int):
        from .graph import ReduceOp
        return Node._reduce_proc(self, dim, ReduceOp.MAX)
        
    ## Data operations derived from the above core ops
    def flatten (self):
        return self.view([math.prod(self.shape)])

    def unsqueeze (self, dim:int):
        if dim < 0:
            dim = len(self.shape) + dim + 1

        d = deepcopy(self.shape)
        d.insert(dim, 1)
        return self.view(d)

    def squeeze (self, dim:int):
        d = deepcopy(self.shape)
        v = d.pop(dim)
        assert v == 1, "Can't squeeze a non-one dim"
        return self.view(d)
    
    def T (self):
        assert len(self.shape) == 2, "Can't call transpose on a non-2-dim tensor"
        return self.permute([1, 0])

    def mT (self): 
        assert len(self.shape) == 3, "Can't call batch transpose on a non-3-dim tensor"
        return self.permute([0, 2, 1])
    
    def mean (self, dim: int):
        return self.sum(dim) / self.shape[dim]
    
    def var (self, dim, correction=0):
        a = (self - self.mean(dim).unsqueeze(dim)).pow2()
        return a.sum(dim) / (self.shape[dim] - correction)
    
    ############################################################
    ## Common Neural Network operations
    # Note that not all NN operations are declared here
    def sigmoid (self): 
        return 1 / (1 + (-self).exp())
    
    def softmax (self, dim:int):
        return self.exp() / self.exp().sum(dim)
    
    # use comparisons; much faster and easier to compute
    def relu (self):
        return (self > 0) * self