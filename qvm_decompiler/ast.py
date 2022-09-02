from abc import ABC, abstractmethod
from .qvm import Opcode


class ASTNode(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other) and [i for i in self] == [i for i in other]


class Expr(ASTNode):
    pass


class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        yield "value", self.value

    def __repr__(self):
        return f"Constant(value={self.value:#x})"


class UnaryOp(Expr):
    def __init__(self, expr):
        self.expr = expr

    def __iter__(self):
        yield "expr", self.expr

    def __repr__(self):
        return f"{self.__class__.__name__}(expr={self.expr!r})"


class Neg(UnaryOp):
    pass


class BCom(UnaryOp):
    pass


class BinOp(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __iter__(self):
        yield "lhs", self.lhs
        yield "rhs", self.rhs

    def __repr__(self):
        return f"{self.__class__.__name__}(lhs={self.lhs!r}, rhs={self.rhs!r})"


class Add(BinOp):
    pass


class Sub(BinOp):
    pass


class Div(BinOp):
    pass


class Mod(BinOp):
    pass


class Mul(BinOp):
    pass


class And(BinOp):
    pass


class Or(BinOp):
    pass


class Xor(BinOp):
    pass


class Lsh(BinOp):
    pass


class Rsh(BinOp):
    pass


class Assign(BinOp):
    pass


class Comparison(BinOp):
    pass


class Eq(Comparison):
    pass


class Ne(Comparison):
    pass


class Lt(Comparison):
    pass


class Le(Comparison):
    pass


class Gt(Comparison):
    pass


class Ge(Comparison):
    pass


class Deref(UnaryOp):
    def __init__(self, size, expr):
        self.size = size
        self.expr = expr

    def __iter__(self):
        yield "size", self.size
        yield from super().__iter__()

    def __repr__(self):
        return f"Deref(size={self.size!r}, expr={self.expr!r})"


class Ref(UnaryOp):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"Ref(expr={self.expr!r})"


class Call(Expr):
    def __init__(self, target, args):
        self.target = target
        self.args = args

    def __iter__(self):
        yield "target", self.target
        yield "args", self.args

    def __repr__(self):
        return f"Call(target={self.target!r}, args={self.args!r})"


class StackVar(Expr):
    def __init__(self, offset):
        self.offset = offset

    def __iter__(self):
        yield "offset", self.offset

    def __repr__(self):
        return f"StackVar(offset={self.offset:#x})"


class Cast(UnaryOp):
    def __init__(self, expr, type):
        super().__init__(expr)
        self.type = type

    def __iter__(self):
        yield from super().__iter__()
        yield "type", self.type

    def __repr__(self):
        return f"Cast(expr={self.expr!r}, type={self.type!r})"


class Stmt(ASTNode):
    pass


class ExprStmt(Stmt):
    def __init__(self, expr):
        self.expr = expr

    def __init__(self):
        yield "expr", self.expr

    def __repr__(self):
        return f"ExprStmt(expr={self.expr!r})"


class Block(Stmt):
    def __init__(self, stmts):
        self.stmts = stmts

    def __iter__(self):
        yield "stmts", self.stmts

    def __repr__(self):
        return f"Block(stmts={self.stmts!r})"


class Goto(Stmt):
    def __init__(self, target):
        self.target = target

    def __iter__(self):
        yield "target", self.target

    def __repr__(self):
        return f"Goto(target={self.target!r})"


class Return(Stmt):
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        yield "value", self.value

    def __repr__(self):
        return f"Return(value={self.value!r})"


class If(Stmt):
    def __init__(self, cond, then_stmt, else_stmt):
        self.cond = cond
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt

    def __iter__(self):
        yield "cond", self.cond
        yield "then_stmt", self.then_stmt
        yield "else_stmt", self.else_stmt

    def __repr__(self):
        return f"If(cond={self.cond}, then_stmt={self.then_stmt}, else_stmt={self.else_stmt})"


class ASTVisitor(ABC):
    def generic_visit(self, node):
        if not isinstance(node, ASTNode):
            return
        for _, child in node:
            if isinstance(child, list):
                for item in child:
                    self.visit(item)
            else:
                self.visit(child)

    def visit(self, node):
        for cls in node.__class__.__mro__:
            if visit_method := getattr(self, f"visit_{cls.__name__}", None):
                return visit_method(node)
        return self.generic_visit(node)


class ASTTransformer(ASTVisitor):
    def generic_visit(self, node):
        if not isinstance(node, ASTNode):
            return node
        for attr, child in node:
            if isinstance(child, list):
                for i, item in enumerate(child):
                    child[i] = self.visit(item)
            else:
                setattr(node, attr, self.visit(child))
        return node


comparison_map = {
    Opcode.EQ: Eq,
    Opcode.NE: Ne,
    Opcode.LTI: Lt,
    Opcode.LEI: Le,
    Opcode.GTI: Gt,
    Opcode.GEI: Ge,
    Opcode.LTU: Lt,
    Opcode.LEU: Le,
    Opcode.GTU: Gt,
    Opcode.GEU: Ge,
    Opcode.EQF: Eq,
    Opcode.NEF: Ne,
    Opcode.LTF: Lt,
    Opcode.LEF: Le,
    Opcode.GTF: Gt,
    Opcode.GEF: Ge,
}

bin_op_map = {
    Opcode.ADD: Add,
    Opcode.SUB: Sub,
    Opcode.DIVI: Div,
    Opcode.DIVU: Div,
    Opcode.MODI: Mod,
    Opcode.MODU: Mod,
    Opcode.MULI: Mul,
    Opcode.MULU: Mul,
    Opcode.BAND: And,
    Opcode.BOR: Or,
    Opcode.BXOR: Xor,
    Opcode.LSH: Lsh,
    Opcode.RSHI: Rsh,
    Opcode.RSHU: Rsh,
    Opcode.ADDF: Add,
    Opcode.SUBF: Sub,
    Opcode.DIVF: Div,
    Opcode.MULF: Mul,
}

unary_op_map = {Opcode.NEGI: Neg, Opcode.NEGF: Neg, Opcode.BCOM: BCom}


def astify(instructions):
    op_stack = []
    ast_nodes = []
    call_args = []
    for instruction in instructions:
        match instruction.op:
            case Opcode.UNDEF | Opcode.IGNORE | Opcode.BREAK:
                pass
            case Opcode.ENTER:
                stack_size = instruction.arg  # TODO: store somewhere or something?
            case Opcode.LEAVE:
                ast_nodes.append(Return(op_stack.pop()))
                stack_size = instruction.arg  # TODO: same question as ENTER
            case Opcode.CALL:
                op_stack.append(Call(op_stack.pop(), call_args))
                call_args = []
            case Opcode.PUSH:
                op_stack.append(None)
            case Opcode.POP:
                ast_nodes.append(op_stack.pop())
                assert len(op_stack) == 0, (hex(instruction.addr), op_stack)  # TODO: is this right?
            case Opcode.CONST:
                op_stack.append(Constant(instruction.arg))
            case Opcode.LOCAL:
                op_stack.append(Ref(StackVar(instruction.arg)))
            case Opcode.JUMP:
                target = op_stack.pop()
                ast_nodes.append(Goto(target))
            case op if op in comparison_map:
                rhs, lhs = op_stack.pop(), op_stack.pop()
                ast_nodes.append(comparison_map[op](lhs, rhs))
            case op if op in bin_op_map:
                rhs, lhs = op_stack.pop(), op_stack.pop()
                op_stack.append(bin_op_map[op](lhs, rhs))
            case (Opcode.LOAD1 | Opcode.LOAD2 | Opcode.LOAD4) as op:
                size = int(2 ** (op - Opcode.LOAD1))
                op_stack.append(Deref(size, op_stack.pop()))
            case (Opcode.STORE1 | Opcode.STORE2 | Opcode.STORE4) as op:
                rhs, lhs = op_stack.pop(), op_stack.pop()
                size = int(2 ** (op - Opcode.STORE1))
                ast_nodes.append(Assign(Deref(size, lhs), rhs))
            case Opcode.ARG:
                arg = op_stack.pop()
                offset = instruction.arg  # TODO: maybe use this one day if needed...
                call_args.append(arg)
            case Opcode.BLOCK_COPY:
                size = instruction.arg
                rhs, lhs = op_stack.pop(), op_stack.pop()
                ast_nodes.append(Assign(Deref(size, lhs), Deref(size, rhs)))
            case Opcode.SEX8:
                # TODO: don't need to add casts if input is already the right type
                # (or have a pass to remove useless casts later)
                op_stack.append(Cast(Cast(op_stack.pop(), "char"), "int"))
            case Opcode.SEX16:
                op_stack.append(Cast(Cast(op_stack.pop(), "short"), "int"))
            case op if op in unary_op_map:
                op_stack.append(unary_op_map[op](op_stack.pop()))
            case Opcode.CVIF:
                op_stack.append(Cast(op_stack.pop(), "float"))
            case Opcode.CVFI:
                op_stack.append(Cast(op_stack.pop(), "int"))
            case op:
                raise Exception(f"Invalid opcode: {op}")
    return ast_nodes
