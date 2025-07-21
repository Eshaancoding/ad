from typing import Callable, List
from .node import Node
from colored import Fore, Style
from .core.print import print_graph as g_print

class Block ():
    def __init__(self):
        self.nodes = []

    def add_to_dep (self, node:Node):
        self.nodes.append(node)

    def remove_from_dep (self, other: Node):
        idx = None
        for i, x in enumerate(self.nodes):
            if x.id_eq(other):
                idx = i
                break
        
        if idx is not None:
            del self.nodes[idx]
            
    def __repr__(self):
        st = ""
        for idx, n in enumerate(self.nodes):
            st += Fore.yellow + str(idx+1) + ") " + Style.reset
            st += n.__repr__()
            st += "\n"
        return st
    
    def print_graph (self):
        g_print(self.nodes)
                
class Context ():
    def __init__ (self):
        self.dep_nodes = []
        self.id = -1
        self.procedure = [Block()] # first procedure is the main block
        self.lock_proc = False
        self.temp_to_expr = {}
        
    def add_to_dep (self, node:Node):
        if not self.lock_proc:
            self.procedure[-1].add_to_dep(node) # always access the last
        
    def remove_from_dep (self, other: Node):
        if not self.lock_proc:
            self.procedure[-1].remove_from_dep(other)
        
    def add_proc (self):
        self.procedure.append(Block())
        
    def pop_proc (self):
        assert len(self.procedure) > 1, "Attempt to pop main procedure!"
        return self.procedure.pop(-1)
        
    def print_graph (self):
        self.procedure[0].print_graph()
       
    # node id tracking
    def get_id (self):
        self.id += 1
        return self.id
    
    def __repr__(self):
        return self.procedure[0].__repr__()
    
    def main_proc (self):
        return self.procedure[0]

context = Context()