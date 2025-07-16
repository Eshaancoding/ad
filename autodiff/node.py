from copy import deepcopy
import math
from typing import List, Callable

class Node:
    ############################################################
    ## Derived methods/init
    # TODO: Add res expr and shape (non-phantom) to super().__init__()
    def __init__(self, children, phantom_shape=None):
        from autodiff import context
        self.temp_id = None # also to be filled out by opt_intermediate within execute

        # as we are creating nodes, we record the latest node being changed within the context
        # as we go through the computation graph, the order of the nodes being changed will also be recorded
        # and being recorded into a procedure
        if phantom_shape is None:
            self.id = context.get_id()
            for ch in children:
                if not isinstance(ch, Node): 
                    raise TypeError("Children is not type of node!")
                context.remove_from_dep(ch)

            context.add_to_dep(self)
        else:
            self.shape = phantom_shape
        
    def bck (self, grad):
        raise NotImplementedError
    
    def __repr__ (self) -> str:
        raise NotImplementedError
    
    ############################################################
    ## Comparisons
    def id_eq (self, other):
        assert isinstance(other, Node), "id eq input invalid"
        return self.id == other.id
    
    def type_eq (self, other): 
        from .phantom import PhantomNode
        if isinstance(other, PhantomNode):
            return other.phantom_type_eq(self)

        assert isinstance(other, Node), "type eq input invalid"
        
        res = (type(self) == type(other)) 
        if self.c_len() != other.c_len():
            return False

        for idx, child in enumerate(self.children()):
            res = (res and child.type_eq(other.c(idx)))
        
        return res
    
    ############################################################
    ## Other methods
    # Calls backend 
    def backward (self):
        from .graph.data.constant import ConstantNode
        self.bck(ConstantNode(1.0, self.shape))
    
    # helper for binary ops
    def _to_node (self, node, dim: List[int]):
        if isinstance(node, int):
            from .graph.data.constant import ConstantNode
            return ConstantNode(float(node), dim)
        elif isinstance(node, float):
            from .graph.data.constant import ConstantNode
            return ConstantNode(node, dim)
        elif not isinstance(node, Node):
            raise TypeError(f"Invalid type {type(node)} when converting to node")
       
        return node 
    
    # helper to iterateover the children of nodes
    def walk (self, f: Callable):
        res = f(self) 
       
        if hasattr(res, "child"):
            res.child = res.child.walk(f)
        elif hasattr(res, "left") and hasattr(res, "right"):
            res.left = res.left.walk(f)
            res.right = res.right.walk(f)
            
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
    ## Binary operations (+, *, -, /)
    def __add__ (self, other):
        from .graph.compute.binary import BinaryNode, BinaryOp
        from .graph.data.broadcast import try_broadcast 
        a, b = try_broadcast(self, self._to_node(other, self.shape))
        return BinaryNode(a, b, BinaryOp.ADD) 
    
    def __radd__ (self, other):
        return self.__add__(other)
    
    def __mul__ (self, other):
        from .graph.compute.binary import BinaryNode, BinaryOp
        from .graph.data.broadcast import try_broadcast 
        a, b = try_broadcast(self, self._to_node(other, self.shape))
        return BinaryNode(a, b, BinaryOp.MULT)
    
    def __rmul__ (self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        from .graph.compute.binary import BinaryNode, BinaryOp
        from .graph.data.broadcast import try_broadcast 
        a, b = try_broadcast(self, self._to_node(other, self.shape) * -1.0)
        return BinaryNode(a, b, BinaryOp.ADD)
    
    def __rsub__(self, other):
        from .graph.compute.binary import BinaryNode, BinaryOp
        from .graph.data.broadcast import try_broadcast 
        a, b = try_broadcast(self._to_node(other, self.shape), self * -1.0)
        return BinaryNode(a, b, BinaryOp.ADD)
    
    def __truediv__(self, other):
        from .graph.compute.binary import BinaryNode, BinaryOp
        from .graph.data.broadcast import try_broadcast 
        a, b = try_broadcast(self, self._to_node(other, self.shape).recip())
        return BinaryNode(a, b, BinaryOp.MULT)
    
    def __rtruediv__(self, other):
        from .graph.compute.binary import BinaryNode, BinaryOp
        from .graph.data.broadcast import try_broadcast 
        a, b = try_broadcast(self.recip(), self._to_node(other, self.shape))
        return BinaryNode(a, b, BinaryOp.MULT)
    
    def __neg__ (self):
        return self.__mul__(-1.0)
    
    ############################################################
    ## Comparison operations
    def __eq__(self, other):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.EQUAL)

    def __lt__(self, other):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.LESS_ZERO)

    def __le__(self, other):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.LESS_OR_EQ_ZERO)

    def __gt__(self, other):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.MORE_ZERO)

    def __ge__(self, other):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self - other, UnaryOp.MORE_OR_EQ_ZERO)
    
    ############################################################
    ## Unary operations
    def exp2 (self):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.EXP2)
    
    def log2 (self):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.LOG2)
    
    def sin (self):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.SIN)
    
    def sqrt (self):
        from .graph.compute.unary import UnaryNode, UnaryOp
        return UnaryNode(self, UnaryOp.SQRT)
    
    def recip (self):
        from .graph.compute.unary import UnaryNode, UnaryOp
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
        from .graph.data.broadcast import BroadcastNode
        assert self.shape[dim] == 1, "Broadcast dim invalid"
        return BroadcastNode(self, dim, size)
    
    def view (self, target_dim: list[int]):
        from .graph.data.view import ViewNode
        # handle -1 dim 
        target_dim = ViewNode.handle_minus_dim(self.shape, target_dim)

        if self.shape != target_dim:
            return ViewNode(self, target_dim)
        else:
            return self
        
    def contigious (self):
        from .graph.data.contigious import ContigiousNode
        return ContigiousNode(self)
    
    def __getitem__(self, idx):
        from .graph.data.index import IndexNode
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
        from .graph.data.permute import PermuteNode
        
        # if contigious list ([0,1,2,..])
        if sum([l[i] == i for i in range(len(l))]) == len(l):
            return self
        else:
            return PermuteNode(self, l)
    
    def _reduce_proc (self, dim: int, op):
        from .graph.compute.reduce import ReduceNode
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
        from .graph.compute.reduce import ReduceOp
        return Node._reduce_proc(self, dim, ReduceOp.SUM)
   
    def max (self, dim: int):
        from .graph.compute.reduce import ReduceOp
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