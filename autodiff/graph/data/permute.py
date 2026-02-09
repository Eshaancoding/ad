from ...node import Node 

class PermuteNode (Node):
    __match_args__ = ("child", "permute_to")
    def __init__(self, child:Node, permute_to: list[int]):
        # Assert 
        assert len(child.shape) == len(permute_to), f"permute to invalid dim {len(child.shape)} {permute_to}"
        for i in permute_to:
            assert i < len(child.shape), f"Invalid permute vec: {permute_to} with child shape: {child.shape}"
        
        # calc shape
        c_dim = child.shape
        dim = [0 for _ in range(len(c_dim))] 
        for i in range(len(c_dim)):
            dim[i] = c_dim[permute_to[i]]

        # init node
        super().__init__([child], dim)
        self.permute_to = permute_to
        
    def _bck(self, grad:Node):
        if not isinstance(grad, Node):
            raise TypeError("Grad is not node!")

        inv_perm = [0 for _ in range(len(self.permute_to))]        
        for i in range(len(self.permute_to)):
            inv_perm[self.permute_to[i]] = i
            
        return grad.permute(inv_perm)
        
    def __repr__ (self):
        return f"{self.id} = Permute {self.permute_to} --> {self.child.id}"

    def repeat_helper (self, is_child):
        return (
            "Permute",
            tuple(self.permute_to),
            self.child.repeat_helper(True) if is_child else ()
        )
