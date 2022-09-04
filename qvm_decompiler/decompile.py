from .ast import *
from .pretty_printer import ASTPrettyPrinter
from .cfg import BasicBlock


class DerefRefSimplifier(ASTTransformer):
    def visit_Deref(self, deref):
        # TODO: size goes away... :(
        deref.expr = self.visit(deref.expr)
        if isinstance(ref := deref.expr, Ref):
            return ref.expr
        return deref


class NodeReplacer(ASTTransformer):
    def __init__(self, tree):
        self.tree = tree
        self.replace_count = 0

    def replace(self, old, new):
        self.old, self.new = old, new
        self.tree = self.visit(self.tree)
        return self.tree, self.replace_count

    def visit_ASTNode(self, node):
        if node == self.old:
            self.replace_count += 1
            return self.new
        self.generic_visit(node)
        return node


def decompile(cfg):
    deref_ref_simplifier = DerefRefSimplifier()
    pretty_printer = ASTPrettyPrinter()
    decompiled_output = ""
    for addr in sorted(cfg.keys()):
        basic_blocks = cfg[addr]

        for block in basic_blocks:
            nodes = block.nodes
            for i, node in enumerate(nodes):
                nodes[i] = deref_ref_simplifier.visit(node)
            for i in range(len(nodes) - 2, -1, -1):  # TODO: Maybe remove if/when we do data flow analysis?
                match nodes[i]:
                    case Assign(lhs=StackVar() as var, rhs=Call() as call):
                        nodes[i + 1], replace_count = NodeReplacer(nodes[i + 1]).replace(var, call)
                        # TODO: Don't like the "<", believe in proper data flow analysis sooner?
                        assert replace_count <= 1, replace_count
                        nodes.pop(i)
            match block:
                case BasicBlock(
                    nodes=[*_, Return(value=value)],
                    successors=[BasicBlock(nodes=[Return(value=None)])],
                ) if value is not None:
                    useless_ret_block = block.successors.pop()
                    useless_ret_block.predecessors.remove(block)

        for block in basic_blocks:
            match block:
                case BasicBlock(
                    successors=[
                        BasicBlock(successors=[post_if], predecessors=[_]) as false_block,
                        BasicBlock(successors=[post_if2], predecessors=[_]) as true_block,
                    ]
                ) if post_if is post_if2:
                    block.nodes[-1] = If(
                        cond=block.nodes[-1],
                        then_stmt=Block(true_block.nodes),
                        else_stmt=Block(false_block.nodes),
                    )
                    block.successors = [post_if]
                    post_if.predecessors.remove(false_block)
                    post_if.predecessors.remove(true_block)
                    post_if.predecessors.append(block)

        for i, block in enumerate(basic_blocks):
            decompiled_output += f"{'block' if i else 'sub'}_{block.id:x}:\n"
            for node in block.nodes:
                decompiled_output += pretty_printer.visit(node) + "\n"
            decompiled_output += f"successors: {[f'block_{succ.id:x}' for succ in block.successors]}\n"
            decompiled_output += "\n"

    return decompiled_output
