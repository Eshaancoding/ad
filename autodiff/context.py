from typing import Callable, List, Optional, Dict, Tuple
from .node import Node
from colored import Fore, Style
from .print_graph import pg
import numpy as np

class Proc ():
    def __init__(self, proc: List[Node]):
        # TODO: Have a check for this hehehe
        # true type: List[Node | FuseBase | AllocCmds]
        self.procedure = proc
        self.id = context.get_proc_id()
        
    def __repr__ (self):
        from .helper import indent_str
        s = ""
        for n in self.procedure:
            s += str(n) + "\n"
            if hasattr(n, "proc"):
                s += indent_str(str(n.get_proc())) + "\n"
        return s
    
    def insert (self, idx, val):
        self.procedure.insert(idx, val)

    def append (self, val):
        self.procedure.append(val)
        
    def walk (self, func: Callable[[Node, int], Optional[Node]], step_fused:bool = True, step_proc:bool = True):
        """Walks over the function to be called. If returned None, then it will delete the node

        Args:
            func (Callable[[Node], Node]): _description_
        """
        from .fusion import FuseBase
        for idx in range(len(self.procedure)):
            n = self.procedure[idx]
            if isinstance(n, FuseBase) and step_fused:
                for n_idx, node in enumerate(n.nodes):
                    self.procedure[idx].nodes[n_idx] = func(node, self.id) 
                self.procedure[idx].nodes = list(filter(lambda x: x is not None, self.procedure[idx].nodes))
            elif isinstance(n, Node) and n.get_proc() is not None and step_proc:
                self.procedure[idx].proc.walk(func, step_fused, step_proc)
            else:
                self.procedure[idx] = func(n, self.id)
                
        self.procedure = list(filter(lambda x: x is not None, self.procedure))

class Block ():
    def __init__(self, id_override=None):
        self.nodes: Dict[int, Tuple[Node, int]] = dict()
        self.position = 0

        # "hacky way" of getting over the context initialization issue
        self.id = id_override if id_override is not None else context.get_block_id()

    def add_to_dep (self, node:Node):
        self.nodes[node.id] = (node, self.position)
        self.position += 1 

    def __len__ (self):
        return len(self.nodes)

    def remove_from_dep (self, other: Node):
        if other.id in self.nodes:
            del self.nodes[other.id] 

    def convert_to_nodes (self):
        if isinstance(self.nodes, List): # if already converted, don't do anything
            return 

        sorted_nodes = sorted(self.nodes.values(), key=lambda kv: kv[1]) # first sort by position
        n = []
        for kv in sorted_nodes:
            n.append(kv[0]) # just get the node for list
        self.nodes = n
            
    def __repr__(self):
        st = ""
        for idx, n in enumerate(self.nodes):
            st += Fore.yellow + str(idx+1) + ") " + Style.reset
            st += n.__repr__()
            st += "\n"
        return st
    
    def print_graph (self):
        pg(self.nodes)

# Shared context as graph is carried out. 
# Includes procedure tracking, dependency list, and unique id generation
class Context ():
    def __init__ (self):
        self.id = -1
        self.proc_id = -1
        self.block_id = 0
        self.program_id = -1
        self.grads_seen = dict()
        self.connections = dict()
        self.conn_seen = set()

        self.procedure = [Block(0)] # first procedure is the main block
        self.lock_proc = False

    # Note dependency tracking as it goes through forward and backward 
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
        ret = self.procedure.pop(-1)
        ret.convert_to_nodes()
        return ret
        
    def print_graph (self):
        self.procedure[0].print_graph()

    # prepares context to execution 
    def prep_exec (self):
        self.lock_proc = True
        self.procedure[0].convert_to_nodes()
       
    # node id tracking
    def get_id (self):
        self.id += 1
        return self.id
    
    def get_proc_id (self):
        self.proc_id += 1
        return self.proc_id 

    def get_block_id (self):
        self.block_id += 1
        return self.block_id

    def get_prog_id (self):
        self.program_id += 1
        return self.program_id

    # alternatives
    def __repr__(self):
        return self.procedure[0].__repr__()
    
    def main_proc (self):
        return self.procedure[0]

context = Context()
