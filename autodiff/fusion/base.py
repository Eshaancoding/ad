from ..node import Node
from ..alloc import AllocEntry
from ..graph import ConstantNode, Tensor
from enum import Enum
from typing import List
from colored import stylize, fore, style
from ..helper import indent_str

"""
There are 4 sets of fusion

Across Layer Fusion -- dependencies across layers (ex: elw --> elw)
Within Layer Fusion --  (ex: elw within an elw with diff dependencies)
Procedure -- Already created internally. Fuses all the layers into a single procedure to be executed step by step
"""

class FuseType (Enum):
    ACROSS_LAYER = 1
    WITHIN_LAYER = 2
    ALL = 3

class FuseBase ():
    def __init__(self, ty: FuseType, is_proc: bool = False):
        from ..context import context
        self.type = ty
        self.nodes: List[Node | AllocEntry] = []
        self.is_proc = is_proc
        self.fuse_id = context.get_fuse_id()

    def _add_node(self, add):
        """
        The below code is for "smartly" trying to insert nodes that has deps at their nearest definition 
        However, this didn't so well. Also weirdly enough, using self.nodes.append works fairly well 
        and accomplishes the same effect
        """

        # if isinstance(add, Tensor) or isinstance(add, ConstantNode):
        # self.nodes.insert(0, add) # also add tensor and constant declarations at the top
        # return
        # match the dependencies
        # deps = get_deps(add)
        # idx = None
        # for i, n in enumerate(self.nodes):
        #     res = get_res(n)
        #     if res in deps:
        #         idx = i+1

        # if idx == None:
        #     self.nodes.insert(0, add)
        # else:
        #     print("IDX SOME!")
        #     self.nodes.insert(idx, add)

        self.nodes.append(add)

    def add(self, n):
        if isinstance(n, FuseBase):
            if not self.is_proc:
                for node in n.nodes:
                    self._add_node(node)
            else:
                if type(self) != type(n):
                    self._add_node(n)
                else:
                    for node in n.nodes:
                        self._add_node(node)
                
        elif isinstance(n, Node):
            self._add_node(n)
        else:
            raise TypeError(f"Invalid type: {type(n)}")

    # node_one and node_two could be Node | FuseBase
    # this has to be implemented by child class
    @staticmethod
    def could_fuse(node_one, node_two):
        raise NotImplementedError(
            "Need to implement could_fuse(); true or false on whether it could fuse")
    
    def __call__(self):
        raise NotImplementedError("Need to derive the base class")

    def __repr__(self):
        st = stylize(f"{type(self).__name__} Fusion", fore("light_blue")) + \
            f" (fuse_id: {stylize(str(self.fuse_id), fore("green") + style("bold"))}):\n"

        for n in self.nodes:
            st += indent_str(str(n)) + "\n"

        return st[:-1]  # remove the last \n
