from abc import ABC, abstractmethod
from enum import IntEnum
from io import BytesIO
from sys import argv

class Opcode(IntEnum):
    UNDEF = 0
    IGNORE = 1
    BREAK = 2
    ENTER = 3
    LEAVE = 4
    CALL = 5
    PUSH = 6
    POP = 7
    CONST = 8
    LOCAL = 9
    JUMP = 10
    EQ = 11
    NE = 12
    LTI = 13
    LEI = 14
    GTI = 15
    GEI = 16
    LTU = 17
    LEU = 18
    GTU = 19
    GEU = 20
    EQF = 21
    NEF = 22
    LTF = 23
    LEF = 24
    GTF = 25
    GEF = 26
    LOAD1 = 27
    LOAD2 = 28
    LOAD4 = 29
    STORE1 = 30
    STORE2 = 31
    STORE4 = 32
    ARG = 33
    BLOCK_COPY = 34
    SEX8 = 35
    SEX16 = 36
    NEGI = 37
    ADD = 38
    SUB = 39
    DIVI = 40
    DIVU = 41
    MODI = 42
    MODU = 43
    MULI = 44
    MULU = 45
    BAND = 46
    BOR = 47
    BXOR = 48
    BCOM = 49
    LSH = 50
    RSHI = 51
    RSHU = 52
    NEGF = 53
    ADDF = 54
    SUBF = 55
    DIVF = 56
    MULF = 57
    CVIF = 58
    CVFI = 59

class ASTNode(ABC):
    @abstractmethod
    def __items__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

class Expr(ASTNode):
    pass
    
class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def __items__(self):
        yield from ()
    
    def __repr__(self):
        return f"Constant(value={self.value:#x})"


class UnaryOp(Expr):
    OP: str

    def __init__(self, target):
        self.target = target

    def __items__(self):
        yield self.target
    
    def __repr__(self):
        return f"{self.__class__.__name__}(target={self.target!r})"

class Neg(UnaryOp):
    OP = '-'

class BCom(UnaryOp):
    OP = '~'

class BinOp(Expr):
    OP: str
    
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __items__(self):
        yield self.lhs
        yield self.rhs
    
    def __repr__(self):
        return f"{self.__class__.__name__}(lhs={self.lhs!r}, rhs={self.rhs!r})"

class Add(BinOp):
    OP = '+'

class Sub(BinOp):
    OP = '-'

class Div(BinOp):
    OP = '/'

class Mod(BinOp):
    OP = '%'

class Mul(BinOp):
    OP = '*'

class And(BinOp):
    OP = '&'

class Or(BinOp):
    OP = '|'

class Xor(BinOp):
    OP = '^'

class Lsh(BinOp):
    OP = '<<'

class Rsh(BinOp):
    OP = '>>'

class Assign(BinOp):
    OP = '='

class Comparison(BinOp):
    pass

class Eq(Comparison):
    OP = "=="

class Ne(Comparison):
    OP = "!="

class Lt(Comparison):
    OP = "<"

class Le(Comparison):
    OP = "<="

class Gt(Comparison):
    OP = ">"

class Ge(Comparison):
    OP = ">="

class Deref(Expr):
    def __init__(self, size, target):
        self.size = size
        self.target = target
    
    def __items__(self):
        yield self.target

    def __repr__(self):
        return f"Deref(size={self.size!r}, target={self.target!r})"

class Ref(Expr):
    def __init__(self, target):
        self.target = target

    def __items__(self):
        yield self.target

    def __repr__(self):
        return f"Ref(target={self.target!r})"

class FunCall(Expr):
    def __init__(self, target, args):
        self.target = target
        self.args = args
    
    def __items__(self):
        yield self.target
        yield from self.args

    def __repr__(self):
        return f"FunCall(target={self.target!r}, args={self.args!r})"

class StackVar(Expr):
    def __init__(self, offset):
        self.offset = offset
    
    def __items__(self):
        yield from ()

    def __repr__(self):
        return f"StackVar(offset={self.offset:#x})"

class Cast(Expr):
    def __init__(self, expr, type):
        self.expr = expr
        self.type = type
    
    def __items__(self):
        yield self.expr
    
    def __repr__(self):
        return f"Cast(expr={self.expr!r}, type={self.type!r})"

class Stmt(ASTNode):
    pass

class ExprStmt(Stmt):
    def __init__(self, expr):
        self.expr = expr

    def __init__(self):
        yield self.expr
    
    def __repr__(self):
        return f"ExprStmt(expr={self.expr!r})"

class Block(Stmt):
    def __init__(self, stmts):
        self.stmts = stmts
    
    def __items__(self):
        yield from self.stmts
    
    def __repr__(self):
        return f"Block(stmts={self.children!r})"

class Goto(Stmt):
    def __init__(self, target):
        self.target = target
    
    def __items__(self):
        yield self.target
    
    def __repr__(self):
        return f"Goto(target={self.target!r})"

class Return(Stmt):
    def __init__(self, value):
        self.value = value
    
    def __items__(self):
        if self.value:
            yield self.value
    
    def __repr__(self):
        return f"Return(value={self.value!r})"
    
def _read_int_32(f):
    return int.from_bytes(f.read(4), "little", signed=True)

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
    Opcode.GEF: Ge
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
    Opcode.MULF: Mul
}

unary_op_map = {
    Opcode.NEGI: Neg,
    Opcode.NEGF: Neg,
    Opcode.BCOM: BCom    
}

with open(argv[1], "rb") as f:
    magic = f.read(4)
    assert magic == b"\x44\x14\x72\x12"
    instruction_count = _read_int_32(f)
    code_offset, code_length = _read_int_32(f), _read_int_32(f)
    data_offset, data_length = _read_int_32(f), _read_int_32(f)
    lit_length, bss_length = _read_int_32(f), _read_int_32(f)

    f.seek(code_offset)
    code = BytesIO(f.read(code_length)) # TODO: this is redundant

    f.seek(data_offset)
    data = f.read(data_length + lit_length)

    op_stack = []
    ast_nodes = []
    call_args = []
    basic_block_leaders = set()
    inst_ast_map = {}
    inst_idx = 0
    def add_node(node, i):
        global inst_idx # TODO idk
        inst_ast_map[inst_idx] = len(ast_nodes)
        inst_idx = i + 1
        ast_nodes.append(node)
    for i in range(instruction_count):
        match code.read(1)[0]:
            case Opcode.UNDEF | Opcode.IGNORE | Opcode.BREAK:
                pass
            case Opcode.ENTER:
                assert len(op_stack) == 0, (hex(i), op_stack)
                if ast_nodes: print(ast_nodes)
                ast_nodes = []
                stack_size = _read_int_32(code)
            case Opcode.LEAVE:
                add_node(Return(op_stack.pop()), i)
                stack_size = _read_int_32(code) # TODO: assert mb
            case Opcode.CALL:
                op_stack.append(FunCall(op_stack.pop(), call_args))
                call_args = []
            case Opcode.PUSH:
                op_stack.append(None)
            case Opcode.POP:
                add_node(op_stack.pop(), i)
                assert len(op_stack) == 0, (hex(i), op_stack) # TODO: is this right?
            case Opcode.CONST:
                op_stack.append(Constant(_read_int_32(code)))
            case Opcode.LOCAL:
                op_stack.append(Ref(StackVar(_read_int_32(code))))
            case Opcode.JUMP:
                target = op_stack.pop()
                if isinstance(target, Constant):
                    basic_block_leaders.add(target.value)
                else:
                    match ast_nodes:
                        case [*_, Lt(lhs=_, rhs=Constant(value=min_bound)), Gt(lhs=_, rhs=Constant(value=max_bound))]:
                            num_entries = max_bound - min_bound + 1
                            match target:
                                case Deref(target=Add(lhs=_, rhs=Constant(value=base))):
                                    base += min_bound * 4
                                    for addr in range(base, base + num_entries * 4, 4):
                                        basic_block_leaders.add(int.from_bytes(data[addr:addr+4], 'little'))
                        case _:
                            raise Exception(f"Unrecognized switch-case @ {i:#x}")
                basic_block_leaders.add(i + 1)
                add_node(Goto(target), i)
            case op if op in comparison_map:
                basic_block_leaders.add(_read_int_32(code))
                basic_block_leaders.add(i + 1)
                rhs, lhs = op_stack.pop(), op_stack.pop()
                add_node(comparison_map[op](lhs, rhs), i)
            case op if op in bin_op_map:
                rhs, lhs = op_stack.pop(), op_stack.pop()
                op_stack.append(bin_op_map[op](lhs, rhs))
            case (Opcode.LOAD1 | Opcode.LOAD2 | Opcode.LOAD4) as op:
                size = int(2**(op - Opcode.LOAD1))
                op_stack.append(Deref(size, op_stack.pop()))
            case (Opcode.STORE1 | Opcode.STORE2 | Opcode.STORE4) as op:
                rhs, lhs = op_stack.pop(), op_stack.pop()
                size = int(2**(op - Opcode.STORE1))
                add_node(Assign(Deref(size, lhs), rhs), i)
            case Opcode.ARG:
                arg = op_stack.pop()
                todo = code.read(1)[0] # TODO: maybe use this one day if needed...
                call_args.append(arg)
            case Opcode.BLOCK_COPY:
                size = _read_int_32(code)
                rhs, lhs = op_stack.pop(), op_stack.pop()
                add_node(Assign(Deref(size, lhs), Deref(size, rhs)), i)
            case Opcode.SEX8:
                # TODO: don't need to add casts if input is already the right type
                # (or have a pass to remove useless casts later)
                op_stack.append(Cast(Cast(op_stack.pop(), 'char'), 'int'))
            case Opcode.SEX16:
                op_stack.append(Cast(Cast(op_stack.pop(), 'short'), 'int'))
            case op if op in unary_op_map:
                op_stack.append(unary_op_map[op](op_stack.pop()))
            case Opcode.CVIF:
                op_stack.append(Cast(op_stack.pop(), 'float'))
            case Opcode.CVFI:
                op_stack.append(Cast(op_stack.pop(), 'int'))
            case op:
                raise Exception(f"Invalid opcode: {op}") # TODO: Find a better class to throw?
    if ast_nodes: print(ast_nodes)
