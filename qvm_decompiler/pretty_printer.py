from .ast import *


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
