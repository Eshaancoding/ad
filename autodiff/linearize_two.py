from copy import deepcopy
from typing import Dict, Set
from autodiff.context import Block, Proc
from autodiff.fusion.helper import get_deps, get_res, print_toposort
from autodiff.graph.control.ir_for import ForNode
from autodiff.helper import walk_graph
from autodiff.node import Node
from autodiff.print_graph import pg
from toposort import toposort
from pprint import pprint

global_deps = {}
id_to_node = {}
block_to_node = {}

def insert_to_global_dep (block_id:int, key:int, value:int):
    global global_deps
    if block_id not in global_deps:
        global_deps[block_id] = dict() # initialize global deps with empty dictionary 

    if key not in global_deps[block_id]:
        if value is not None:
            global_deps[block_id][key] = set({value}) 
        else:
            global_deps[block_id][key] = set({}) 
    elif value is not None:
        global_deps[block_id][key].add(value)

def get_deps (block: Block, alr_defined: Dict[int, int] = dict()):
    global block_to_node
    print("********************* GET DEPS **************")

    defined = dict()
    vars_need_dep = dict()

    for n in block.nodes:
        # step through the block
        if (inner_bl := n.get_block()):
            def_block, need_dep_block = get_deps(inner_bl, alr_defined | defined)
        
            for (dep, block_id) in deepcopy(list(need_dep_block.items())):
                if block_id == block.id:
                    insert_to_global_dep(block.id, n.id, dep)
                    del need_dep_block[dep]

            # update variables
            defined.update(def_block)
            vars_need_dep.update(need_dep_block)
            block_to_node[inner_bl.id] = n.id
        else:
            # if regular node, go through the ids
            def update_node (n:Node):
                # skip if node if already defined
                if n.id in alr_defined:
                    return n

                if isinstance(n, Node):
                    # insert to global deps
                    for dep in n.kargs_child_ids():
                        should_visit = True 
                        key = n.id
                        value = dep

                        # referencing node already defined by lower-level blocks
                        # ex: for loop referencing tensor defined in global block
                        if dep in alr_defined:
                            value = None
                            should_visit = False
                            vars_need_dep[dep] = alr_defined[dep]
                        
                        # referencing ndoe already defined by upper-level blocks
                        # ex: Receiver of the latest weight changed by upper for loop 
                        if dep in defined:
                            if (block_id_defined := defined[dep]) != block.id:
                                should_visit = False
                                value = block_to_node[block_id_defined]

                        # insert to global dep
                        insert_to_global_dep(block.id, key, value)

                        # visit
                        if should_visit:
                            update_node(id_to_node[dep])

                    # add to defined if not inserted already 
                    if n.id not in defined:
                        defined[n.id] = block.id
                return n
            
            update_node(n)

    return defined, vars_need_dep

def fuse (g_dep_id) -> Proc:
    # get deps
    global global_deps
    deps = global_deps[g_dep_id]

    # run through toposort
    toposort_res = list(toposort(deps))
    print_toposort(toposort_res, id_to_node)

    ########### Fusing Operation ########### 

    # Linearize --> return as Proc
    proc = []
    for layer in toposort_res:
        for node_id in layer:
            node = id_to_node[node_id]
            # if we have a node with block (ex: for), then fuse that as well
            # should already exist in g_deps
            if (block := node.get_block()) is not None:
                node.proc = fuse(block.id)
            proc.append(node) 

    return Proc(proc)

def linearize_two (main_block: Block):
    global global_deps, id_to_node, block_to_node

    ######## Insert all id to node ######## 
    id_to_node = {}
    def insert_id_to_node (node: Node, _):
        if node.id not in id_to_node:
            id_to_node[node.id] = node
        return node
    walk_graph(main_block, insert_id_to_node, walk_block=True, walk_child=True) 
    
    ######## Get deps ########
    global_deps = {}
    block_to_node = {}
    _, d = get_deps(main_block)
    assert len(d) == 0, "vars need dep not None!"
    pprint(global_deps)

    ######## Given the deps: fuse ########
    return fuse(0) # fuse the global node + return proc 
     
