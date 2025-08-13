# Like feeder, this would receive data. Once available, it would call the python function
from typing import Callable, List
from ...node import Node

class Receiver (Node): 
    __match_args__ = ("func", "nodes", "name")
    def __init__ (self, func:Callable, rec_children: List[Node], name=""):
        super().__init__(rec_children, [], None, is_receiver=True)
        self.func = func
        self.name = name

    def bck (self, _:Node):
        pass # no backward on receiver
        
    def __repr__ (self) -> str:
        st = ", ".join([str(karg) for karg in self.kargs])
        return f"{self.id} = Receiver \"{self.name}\" --> ({st})"

    def node_eq (self, other) -> bool:
        if not isinstance(other, Receiver):
            return False

        # almost always false. Receiver node shouldn't be combined often
        return self.id == other.id
