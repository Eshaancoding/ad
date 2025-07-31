from typing import Callable, List, Optional
from .node import Node
from colored import Fore, Style
from .print import print_graph as g_print

class Proc ():
    def __init__(self, proc: List[Node]):
        # TODO: Have a check for this hehehe
        # true type: List[Node | FuseBase | AllocCmds]
        self.procedure = proc
        self.id = context.get_proc_id()
        
    def __repr__ (self):
        from .helper import indent
        s = ""
        for n in self.procedure:
            s += str(n) + "\n"
            if hasattr(n, "proc"):
                s += indent(str(n.get_proc()), prefix=r'    ') + "\n"
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
    def __init__(self):
        self.nodes: List[Node] = []

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

# Shared context as graph is carried out. 
# Includes procedure tracking, dependency list, and unique id generation
class Context ():
    def __init__ (self):
        self.dep_nodes = []

        self.id = -1
        self.fuse_id = -1
        self.proc_id = -1
        self.program_id = -1

        self.procedure = [Block()] # first procedure is the main block
        self.lock_proc = False
        self.temp_to_expr = {}

        self.deps = []         # add dependency list 
        self.lenient_dep = False 

    def add_dep_list (self, id:str):
        self.deps.append(id)
        
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
    
    def get_fuse_id (self):
        self.fuse_id += 1
        return self.fuse_id

    def get_proc_id (self):
        self.proc_id += 1
        return self.proc_id 

    def get_prog_id (self):
        self.program_id += 1
        return self.program_id
    
    def __repr__(self):
        return self.procedure[0].__repr__()
    
    def main_proc (self):
        return self.procedure[0]

context = Context()
