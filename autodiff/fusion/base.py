from ..node import Node
from typing import List, Set
from colored import stylize, fore, style
from ..helper import indent_str
from autodiff.fusion.helper import get_deps, get_res
from enum import Enum
    
class FuseResult (Enum):
    INIT=1
    ADD=2
    NONE=3

class FuseBase ():
    def __init__(
            self, 
            dep_match=True,
            should_flatten=False
        ):

        from ..context import context
        self.dep_match = dep_match
        self.should_flatten = should_flatten

        self.nodes: List[Node] = []
        self.init_node = None
        self.fuse_id = context.get_id()

    def _add_node(self, add, index=None):
        if isinstance(index, int): 
            self.nodes.insert(index, add)
        else:
            self.nodes.append(add)

    def add(self, n, index=None):
        if isinstance(n, FuseBase):
            if self.should_flatten and type(self) != type(n):
                self._add_node(n, index)
            else:
                for node in n.nodes:
                    self._add_node(node, index)
        elif isinstance(n, Node):
            self._add_node(n, index)
        else:
            raise TypeError(f"Invalid type: {type(n)}")

    def _fuse (self, node_one, node_two):
        raise NotImplementedError("Need to implement could_fuse(); true or false on whether node_one and node_two could fuse")

    def _init (self, node):
        raise NotImplementedError("Need to implement _init(); true or false on whether ") 
        
    # use these function at merge
    def attempt_fuse (self, node) -> FuseResult:
        # if we are at 0 nodes
        if len(self.nodes) == 0 and self._init(node):
            self.init_node = node
            self.add(node)
            return FuseResult.INIT

        results = get_res(self)
        
        # check if any of the nodes can fuse with the current node
        for idx, n in enumerate(reversed(self.nodes)):
            # if karg_match, then check res and deps match
            if self.dep_match:
                deps = get_deps(node)
                if len(results.intersection(deps)) == 0:
                    continue  

            # Then, check if we can fuse
            if self._fuse(n, node):
                self.add(node, len(self.nodes)-idx)
                return FuseResult.ADD

        return FuseResult.NONE

    # For debugging purposes
    def __repr__(self):
        st = stylize(f"{type(self).__name__} Fusion", fore("light_blue")) + \
            f" (fuse_id: {stylize(str(self.fuse_id), fore("green") + style("bold"))}):\n"

        for n in self.nodes:
            st += indent_str(str(n)) + "\n"

        return st[:-1]  # remove the last \n
