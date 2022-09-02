from sys import argv

from .ast import *
from .cfg import BasicBlock, build_cfg
from .qvm import Qvm


class DerefRefSimplifier(ASTTransformer):
    def visit_Deref(self, deref):
        # TODO: size goes away... :(
        deref.expr = self.visit(deref.expr)
        if isinstance(ref := deref.expr, Ref):
            return ref.expr
        return deref


class ASTPrettyPrinter(ASTVisitor):
    def get_precedence(self, expr):  # TODO: ????
        match expr:
            case Ref() | Deref() | Cast() | Neg() | BCom():
                return 2
            case Div() | Mod() | Mul():
                return 3
            case Add() | Sub():
                return 4
            case Lsh() | Rsh():
                return 5
            case Lt() | Le() | Gt() | Ge():
                return 6
            case Eq() | Ne():
                return 7
            case And():
                return 8
            case Xor():
                return 9
            case Or():
                return 10
            case Assign():
                return 14
            case Constant() | StackVar() | Call():
                return 0
            case _:
                raise Exception(f"Unrecognized: ({expr!r})")

    def get_associativity(self, expr):  # TODO: ????
        match expr:
            case Ref() | Deref() | Cast() | Neg() | BCom():
                return "RTL"
            case Div() | Mod() | Mul() | Add() | Sub() | Lsh() | Rsh() | Lt() | Le() | Gt() | Ge() | Eq() | Ne() | And() | Xor() | Or():
                return "LTR"
            case Assign():
                return "RTL"
            case Constant() | StackVar():
                return
            case _:
                raise Exception(f"Unrecognized: ({expr!r})")

    def visit_ASTNode(self, node):
        raise Exception(f"Unhandled ASTNode: {node!r}")

    def visit_Constant(self, node):
        return f"{node.value:#x}"

    def visit_StackVar(self, node):
        return f"local_{node.offset:x}"  # TODO: args vs locals

    def visit_Call(self, node):
        target = node.target
        if isinstance(target, Constant):
            if (value := target.value) < 0:
                target = f"trap_{target.value & 0xFFFFFFFF:x}"
            else:
                target = f"sub_{target.value:x}"
        else:
            target = self.visit(target)
        return f"{target}({', '.join(self.visit(arg) for arg in node.args)})"

    def visit_UnaryOp(self, node):
        match node:
            case BCom():
                op = "~"
            case Neg():
                op = "-"
            case Ref():
                op = "&"
            case Deref():
                op = "*"
            case Cast():
                op = f"({node.type})"
            case _:
                raise Exception(f"Unrecognized UnaryOp: ({node!r})")
        node_prec, node_assoc = self.get_precedence(node), self.get_associativity(node)
        expr = self.visit(node.expr)
        expr_prec = self.get_precedence(node.expr)
        if expr_prec > node_prec or (expr_prec == node_prec and node_assoc == "LTR"):
            expr = f"({expr})"
        return f"{op}{expr}"

    def visit_BinOp(self, node):
        match node:
            case Add():
                op = "+"
            case Sub():
                op = "-"
            case Div():
                op = "/"
            case Mod():
                op = "%"
            case Mul():
                op = "*"
            case And():
                op = "&"
            case Or():
                op = "|"
            case Xor():
                op = "^"
            case Lsh():
                op = "<<"
            case Rsh():
                op = ">>"
            case Eq():
                op = "=="
            case Ne():
                op = "!="
            case Lt():
                op = "<"
            case Le():
                op = "<="
            case Gt():
                op = ">"
            case Ge():
                op = ">="
            case Assign():
                op = "="
            case _:
                raise Exception(f"Unrecognized BinOp: ({node!r})")

        node_prec, node_assoc = self.get_precedence(node), self.get_associativity(node)
        lhs = self.visit(node.lhs)
        lhs_prec = self.get_precedence(node.lhs)
        if lhs_prec > node_prec or (lhs_prec == node_prec and node_assoc == "RTL"):
            lhs = f"({lhs})"

        rhs = self.visit(node.rhs)
        rhs_prec = self.get_precedence(node.rhs)
        if rhs_prec > node_prec or (rhs_prec == node_prec and node_assoc == "LTR"):
            rhs = f"({rhs})"

        return f"{lhs} {op} {rhs}"

    def visit_Goto(self, node):
        target = self.visit(node.target)
        return f"goto {target};"

    def visit_Return(self, node):
        if node.value is not None:
            value = self.visit(node.value)
            return f"return {value};"
        return "return;"

    def visit_If(self, node):  # TODO: else if
        cond = self.visit(node.cond)
        then_stmt = self.visit(node.then_stmt)
        else_stmt = self.visit(node.else_stmt)
        return f"if({cond}) { then_stmt } else { else_stmt }"

    def visit_Block(self, node):
        # TODO: let statements handle their own semicolons
        return "{\n\t" + ";\n\t".join(self.visit(stmt) for stmt in node.stmts) + ";\n}"


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


qvm = Qvm(argv[1])
cfg = build_cfg(qvm)

deref_ref_simplifier = DerefRefSimplifier()

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

    pretty_printer = ASTPrettyPrinter()
    for i, block in enumerate(basic_blocks):
        print(f"{'block' if i else 'sub'}_{block.id:x}:")
        for node in block.nodes:
            print(pretty_printer.visit(node))
        print("successors:", [f"block_{succ.id:x}" for succ in block.successors])
        print()
