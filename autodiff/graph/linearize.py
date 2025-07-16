from ..context import Context
from ..node import Node
from toposort import toposort

def linearize (context: Context):
    
    id_to_node = {}
    def test_toposort(n: Node, g_dep):
        n_id = n.id
        id_to_node[n_id] = n
        ids_dep = [child.id for child in n.children()]

        if n_id not in g_dep:
            g_dep[n_id] = list(set(ids_dep))
        else:
            g_dep[n_id].extend(ids_dep)
            g_dep[n_id] = list(set(g_dep[n_id]))
        
        for child in n.children():
            test_toposort(child, g_dep)

        return n

    g_dep = {}
    context.apply_per_node(lambda n: test_toposort(n, g_dep))
    print(g_dep)
    res = list(toposort(g_dep))
    
    for layer in res:
        print("\n=================== LAYER ====================")
        for i in layer:
            print(id_to_node[i].format_single())