# Like feeder, this would receive data. Once available, it would call the python function
from typing import Callable, List

from colored import fore, stylize
from ...node import Node

class Receiver (Node): 
    __match_args__ = ("func", "nodes", "name")
    def __init__ (self, func:Callable, rec_children: List[Node], name=""):
        super().__init__(rec_children, [], None, is_receiver=True)
        self.func = func
        self.name = name

    def _bck (self, _:Node):
        pass # no backward on receiver
        
    def __repr__ (self) -> str:
        st = ", ".join([str(karg) for karg in self.kargs])
        op_str = stylize("Receiver", fore("turquoise_2"))
        return f"{self.id} = {op_str} \"{self.name}\" --> ({st})"
    
    def repeat_helper(self, is_child):
        return (
            "Receiver",
            self.id
        )
