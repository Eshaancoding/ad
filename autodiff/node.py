from copy import deepcopy
import math
from typing import Any, List, Optional, Tuple
from enum import Enum
from .kernalize import NoneKernelArg, KernelArg

class ChildrenType (Enum):
    SOURCE=0
    BINARY=1
    UNARY=2
    RECEIVER=3

class Node:
    ############################################################
    
    ## Derived methods/init
    # TODO: Add res expr and shape to super().__init__()
    def __init__(self, children, shape:List[int], id_override:Optional[int] = None, is_receiver=False):
        from autodiff import context

        self.id = context.get_id() if id_override is None else id_override
        
        self.children_shapes: List[List[int]] = []
        self.children_datacmds: Optional[List[List[Node]]] = [] # will be only datacmds nodes
        self.kargs: List[KernelArg] = []
        self.kres: KernelArg = NoneKernelArg() # result of the kernel argumnet; Filled at kernalize
        for ch in children:
            if not isinstance(ch, Node): 
                raise TypeError("Children is not type of node!")
            
            # this shape could change (especially at kernalize). Save the deepcopy of shape and use this shape
            # One tensor can be represented in multiple dimensions/views (depends on the children exprs)
            self.children_shapes.append(deepcopy(ch.shape))

            # fill in children datacmds. This is an internal variable needed at kernalize
            self.children_datacmds.append([])

            # fill in kargs. Filled in at kernalize (uses children_datacmds)
            self.kargs.append(NoneKernelArg())

            # as we are creating nodes, we record the latest node being changed within the context
            # as we go through the computation core, the order of the nodes being changed will also be recorded
            # and being recorded into a procedure
            if not is_receiver:
                context.remove_from_dep(ch)
        
        context.add_to_dep(self)

        self.val = None # for representing output after execution

        # set the self.left, self.right, or self.child at children
        # It's easier to seperate them like this rather than having one single list[Node]
        self.children_type = ChildrenType.SOURCE
        if is_receiver:
            self.rec_children = children # only in the case of receiver node  
            self.children_type = ChildrenType.RECEIVER
        elif len(children) == 1:
            self.child: Node = children[0]
            self.children_type = ChildrenType.UNARY
        elif len(children) == 2:
            self.left: Node = children[0]
            self.right: Node = children[1]
            self.children_type = ChildrenType.BINARY
        elif len(children) != 0:
            raise TypeError("Children length is invalid")

        # helpful for backward propogation
        self.incoming_grad = None
        self.shape = list(shape) # set shape
        
    def __repr__ (self) -> str:
        raise NotImplementedError

    def repeat_helper (self, is_child=False):
        raise NotImplementedError

    ############################################################
    ## Backward Propogation
    # returns nodes related to backward computation
    def _bck (self, grad):
        raise NotImplementedError

    def _record_connections (self, parent_id=None, visited=set()): 
        """
        This function tracks the connections of the computation graph
        This is extremely helpful as we do not want to go backward node
        until we have reached all accumulated gradients 
        """
        from .context import context
        visited.add(self.id)

        if parent_id is not None:
            if self.id not in context.connections:
                context.connections[self.id] = [parent_id]
            else:
                context.connections[self.id].append(parent_id)

        for child in self.children():
            if child.id not in visited: 
                child._record_connections(
                    parent_id=self.id, 
                    visited=visited
                )
            else:
                context.connections[child.id].append(self.id)
                

    # Every node will have an incoming gradient. Every incoming gradient
    # are accumulated before it goes through children
    def backward (self, grad=None):
        from .graph import ConstantNode 
        from .context import context
        from autodiff.graph.compute.binary import BinaryNode, BinaryOp

        # running backward globally
        if grad is None:
            # reset context
            context.grads_seen = dict()  
            context.connections = dict()
            context.conn_seen = set()
            self._record_connections(parent_id=None, visited=set())
            grad = ConstantNode(1.0, self.shape)

        # check if its in context seen    
        # if seen, then add grad accumulation
        if self.id in context.grads_seen:
            self.incoming_grad = BinaryNode(
                self.incoming_grad,
                grad,
                BinaryOp.ADD,
                False
            )

            context.grads_seen[self.id] += 1
        else:
            # add to grads seen, update incoming grad
            context.grads_seen[self.id] = 1
            self.incoming_grad = grad

        # else, go through children backward only if we have accumulated all incoming gradients
        num_connect = len(context.connections[self.id]) if self.id in context.connections else 1 
        if num_connect == context.grads_seen[self.id]:
            grad_result = self._bck(self.incoming_grad)

            if self.children_type == ChildrenType.UNARY:
                if grad_result is not None: 
                    self.child.backward(grad_result)
            elif self.children_type == ChildrenType.BINARY: 
                if grad_result[0] is not None: 
                    self.left.backward(grad_result[0])
                if grad_result[1] is not None: 
                    self.right.backward(grad_result[1])
            

    ############################################################
    ## Other methods
    # Calls backend 
    def __hash__ (self):
        return hash(self.id)

    def keep (self):
        """
        Adds the node into dep list
        """
        from .context import context
        context.add_dep_list(self)
    
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
    
    # helper to get children 
    # note that if this function is changed, make sure that the helper function walk_node is changed accordingly
    def children (self):
        from autodiff.graph import Receiver
        if isinstance(self, Receiver):
            return self.rec_children
        elif hasattr(self, "left") and hasattr(self, "right"):
            return [self.left, self.right]
        elif hasattr(self, "child"):
            return [self.child]
        else:
            return []
 
    # gets the children ids from kwargs (used often at linearize, after kernalize operation is done)
    def kargs_child_ids (self, filter_temp=False):
        r = []
        for k in self.kargs:
            r.extend(k.get_ids(filter_temp)) 
        return list(set(r))

    # test on whether node are equal to each other
    def id_eq (self, other):
        assert isinstance(other, Node), "id eq input invalid"
        return self.id == other.id

    # get whether node has a block
    def get_block (self):
        if hasattr(self, "block"):
            from .context import Block
            if isinstance(self.block, Block):
                return self.block
        return None
    
    # get whether node has a proc (used after linearize)
    def get_proc (self):
        if hasattr(self, "proc"):
            from .context import Proc
            if isinstance(self.proc, Proc):
                return self.proc
        return None
    
    # rename the kargs inputs and id (ideally, should be hidden by user); at kernalize
    def rename (self, fr:int, to:int):
        for idx in range(len(self.kargs)):
            self.kargs[idx].rename(fr, to)
        if not self.kres.is_none(): # Tensor kres can be None
            self.kres.rename(fr, to)        

        # replace id
        if self.id == fr:
            self.id = to
            
    # set to temp (ideally, should be hidden from user); at kernalize
    def change_to_temp (self, search:int):
        for idx in range(len(self.kargs)):
            self.kargs[idx].change_to_temp(search)
        if not self.kres.is_none(): # Tensor kres can be None
            self.kres.change_to_temp(search)

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

    # in place operations +=, -=, *=, /=
    def __iadd__ (self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self, self._to_node(other, self.shape), only_right_broadcast=True)
        return BinaryNode(a, b, BinaryOp.ADD, in_place=True)
 
    def __isub__ (self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self, self._to_node(other, self.shape) * -1.0, only_right_broadcast=True)
        return BinaryNode(a, b, BinaryOp.ADD, in_place=True)
 
    def __imult__ (self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self, self._to_node(other, self.shape), only_right_broadcast=True)
        return BinaryNode(a, b, BinaryOp.MULT, in_place=True)
 
    def __itruediv__ (self, other):
        from .graph import BinaryNode, BinaryOp, try_broadcast
        a, b = try_broadcast(self, self._to_node(other, self.shape).recip(), only_right_broadcast=True)
        return BinaryNode(a, b, BinaryOp.MULT, in_place=True)

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
    # Data Manipulation operations
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
    
    def var (self, dim, correction=1):
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
