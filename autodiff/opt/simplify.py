from autodiff.context import Context, context
from autodiff.graph.compute.binary import BinaryNode, BinaryOp
from autodiff.graph.compute.unary import UnaryNode, UnaryOp
from autodiff.graph.data.constant import ConstantNode
from autodiff.node import Node
from autodiff.helper import walk_graph
import math
from typing import Dict

is_new = False

def _inner_simpl (node: Node, visited: Dict[int, int], on_child=False):
    global is_new, simpl_nodes
    if node.id in visited:
        return node

    match node:
        ############ Constant simplification ###############
        case BinaryNode(
            ConstantNode(_ as c1, _ as sh1),
            ConstantNode(_ as c2, _ as sh2),
            _ as op,
            _
        ):
            assert sh1 == sh2, "shape not equal at simplify constant"
            is_new = True
            ret = ConstantNode(c1 + c2 if op == BinaryOp.ADD else c1 * c2, shape=sh1)
            context.add_dep_replace(node.id, ret.id) 
            return ret

        ############ n + 0.0 = n ###############
        case BinaryNode(
            _ as n,
            ConstantNode(0.0, sh),
            _ as op,
            _
        ):
            is_new = True
            if op == BinaryOp.ADD:
                context.add_dep_replace(node.id, n.id)
                return n
            elif op == BinaryOp.MULT:
                r = ConstantNode(0.0, sh)
                context.add_dep_replace(node.id, r.id)
                return r

        case BinaryNode(
            ConstantNode(0.0, sh),
            _ as n,
            _ as op,
            _
        ):
            is_new = True
            if op == BinaryOp.ADD:
                context.add_dep_replace(node.id, n.id)
                return n
            elif op == BinaryOp.MULT:
                return ConstantNode(0.0, sh)

        ############ n * 1.0 = n ###############
        case BinaryNode(
            _ as n,
            ConstantNode(1.0, sh),
            BinaryOp.MULT,
            _
        ):
            is_new = True
            context.add_dep_replace(node.id, n.id)
            return n

        case BinaryNode(
            ConstantNode(1.0, sh),
            _ as n,
            BinaryOp.MULT,
            _
        ):
            is_new = True
            context.add_dep_replace(node.id, n.id)
            return n

        ################ Unary simplfication of constant ################
        case UnaryNode(
            ConstantNode(_ as c, _ as sh),
            _ as unary_op
        ):
            is_new = True
            result_val = None
            match unary_op:
                case UnaryOp.EXP2:
                    result_val = math.exp2(c)
                case UnaryOp.LOG2:
                    result_val = math.log2(c)
                case UnaryOp.SIN:
                    result_val = math.sin(c)
                case UnaryOp.RECIP:
                    result_val = 1.0 / c
                case UnaryOp.SQRT:
                    result_val = math.sqrt(c)
                case UnaryOp.EQUAL:
                    result_val = float(c == 0.0)
                case UnaryOp.MORE_ZERO:
                    result_val = float(c > 0.0)
                case UnaryOp.LESS_ZERO:
                    result_val = float(c < 0.0)
                case UnaryOp.MORE_OR_EQ_ZERO:
                    result_val = float(c >= 0.0)
                case UnaryOp.LESS_OR_EQ_ZERO:
                    result_val = float(c <= 0.0)

            ret = ConstantNode(result_val, sh)
            context.add_dep_replace(node.id, ret.id)
            return ret

    # if on parent node, reset the visited and is_new for going through the child
    if not on_child:
        is_new = False

    # iterate over children
    if hasattr(node, "left") and hasattr(node, "right"):
        node.left = _inner_simpl(node.left, visited, on_child=True)
        node.right = _inner_simpl(node.right, visited, on_child=True)
    elif hasattr(node, "child"):
        node.child = _inner_simpl(node.child, visited, on_child=True)
    
    # if on parent node, ensure that it is constantly being simplified until we can't anymore (is_new = False)
    if not on_child:
        while is_new:
            visited = {}
            is_new = False
            if hasattr(node, "left") and hasattr(node, "right"):
                node.left = _inner_simpl(node.left, visited, on_child=True)
                node.right = _inner_simpl(node.right, visited, on_child=True)
            elif hasattr(node, "child"):
                node.child = _inner_simpl(node.child, visited, on_child=True)
            else:
                break

    visited[node.id] = 1 

    return node


def simpl_node (context: Context):
    # internally, walk 
    context.procedure[0] = walk_graph(context.procedure[0], _inner_simpl, walk_block=True, walk_child=False)
