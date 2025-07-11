from ...node import Node 
from ..helper import indent

class PermuteNode (Node):
    def __init__(self, child:Node, permute_to: list[int]):
        super().__init__([child])
        
        assert len(child.shape()) == len(permute_to), f"permute to invalid dim {len(child.shape())} {permute_to}"

        for i in permute_to:
            assert i < len(child.shape()), f"Invalid permute vec: {permute_to} with child shape: {child.shape()}"
        
        self.permute_to = permute_to
        
    def shape (self):
        c_dim = self.child().shape()
        dim = [0 for _ in range(len(c_dim))] 
        for i in range(len(c_dim)):
            dim[i] = c_dim[self.permute_to[i]]
        return dim
    
    def bck(self, grad:Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not node!")

        inv_perm = [0 for _ in range(len(self.permute_to))]        
        for i in range(len(self.permute_to)):
            inv_perm[self.permute_to[i]] = i
            
        self.child().bck(grad.permute(inv_perm))
        
    def __repr__ (self):
        return f"Permute {self.permute_to}\n{indent(self.child().__repr__())}"