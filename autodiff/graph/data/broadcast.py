from ...node import Node
from ..helper import indent
from copy import deepcopy

#########################################
## Broadcast node
class BroadcastNode (Node):
    def __init__ (self, child: Node, dim: int, size: int):
        super().__init__([child])
        
        self.dim = dim
        self.size = size 
        
        # data cmds doesn't implement the res_expr 
        # however, does implement shape
        p = deepcopy(self.child().shape)
        p[self.dim] = self.size
        self.shape = p
        
    def bck (self, grad:Node):
        if not isinstance (grad, Node):
            raise TypeError("Grad is not node!")
        
        self.child().bck(grad.sum(self.dim))
        
    def __repr__ (self) -> str:
        return f"Broadcast dim {self.dim} to size {self.size}\n{indent(self.child().__repr__())}"
        
###########################################
## Functions to automatically create broadcast node when needed

def is_broadcastable(dim_a: list[int], dim_b: list[int]) -> bool:
    if not dim_a or not dim_b:
        return False

    max_dim = max(len(dim_a), len(dim_b))

    for i in range(1, max_dim + 1):
        a_val = dim_a[-i] if i <= len(dim_a) else 1
        b_val = dim_b[-i] if i <= len(dim_b) else 1

        if a_val != b_val and a_val != 1 and b_val != 1:
            return False
    return True

def make_broadcast_node(n: Node, target_dim: list[int]) -> Node:
    n_dim = deepcopy(n.shape)
    assert len(target_dim) >= len(n_dim), "Wrong broadcasted node"

    # Pad with 1s
    for _ in range(len(target_dim) - len(n_dim)):
        n_dim.insert(0, 1)

    # Reshape the tensor to match broadcast shape
    ret_n = n.view([int(i) for i in n_dim])
    for i in range(len(target_dim)):
        if target_dim[i] != n_dim[i]:
            assert n_dim[i] == 1, "Can't broadcast to target dim"
            ret_n = ret_n.broadcast(i, target_dim[i])

    return ret_n

def try_broadcast(a: Node, b: Node) -> tuple[Node, Node]:
    from .constant import ConstantNode

    # Normalize scalar tensors
    if len(a.shape) == 0 and len(b.shape) > 0:
        a = a.unsqueeze(0)
    if len(b.shape) == 0 and len(a.shape) > 0:
        b = b.unsqueeze(0)

    if a.shape == b.shape:
        return a, b

    if isinstance(a, ConstantNode) or isinstance(b, ConstantNode):
        return a, b

    if is_broadcastable(a.shape, b.shape):
        # Prefer broadcasting smaller/shallower tensor
        a_dim_len = len(a.shape)
        b_dim_len = len(b.shape)

        if a_dim_len == b_dim_len:
            is_a_broadcast = sum(a.shape) < sum(b.shape)
        else:
            is_a_broadcast = a_dim_len < b_dim_len

        if is_a_broadcast:
            return make_broadcast_node(a, b.shape), b
        else:
            return a, make_broadcast_node(b, a.shape)

    raise ValueError(f"Cannot broadcast {a.shape} to {b.shape}")
