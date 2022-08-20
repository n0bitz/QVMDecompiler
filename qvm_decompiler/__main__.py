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
    children: list #[ASTNode]?

    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass

class Expr(ASTNode):
    pass
    
class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"{self.value:#x}"
    
    def __repr__(self):
        return f"Constant(value={self.value:#x})"


class UnaryOp(Expr):
    OP: str

    def __init__(self, target):
        self.children = [target]

    @property
    def target(self):
        return self.children[0]

    def __str__(self):
        return f"{self.OP}{stringify_expr(self.target)}"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(target={self.target!r})"

class NegI(UnaryOp):
    OP = '-'

class NegF(UnaryOp):
    OP = '-'

class BCom(UnaryOp):
    OP = '~'

class BinOp(Expr):
    OP: str
    
    def __init__(self, lhs, rhs):
        self.children = [lhs, rhs]

    @property
    def lhs(self):
        return self.children[0]
    
    @property
    def rhs(self):
        return self.children[1]

    def __str__(self):
        return f"{stringify_expr(self.lhs)} {self.OP} {stringify_expr(self.rhs)}"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(lhs={self.lhs!r}, rhs={self.rhs!r})"

class Add(BinOp):
    OP = '+'

class Sub(BinOp):
    OP = '-'

class DivI(BinOp):
    OP = '/'

class DivU(BinOp):
    OP = '/'

class ModI(BinOp):
    OP = '%'

class ModU(BinOp):
    OP = '%'

class MulI(BinOp):
    OP = '*'

class MulU(BinOp):
    OP = '*'

class And(BinOp):
    OP = '&'

class Or(BinOp):
    OP = '|'

class Xor(BinOp):
    OP = '^'

class Lsh(BinOp):
    OP = '<<'

class RshI(BinOp):
    OP = '>>'

class RshU(BinOp):
    OP = '>>'

class AddF(BinOp):
    OP = '+'

class SubF(BinOp):
    OP = '-'

class DivF(BinOp):
    OP = '/'

class MulF(BinOp):
    OP = '*'

class Assign(BinOp):
    OP = '='

class Comparison(BinOp):
    pass

class Eq(Comparison):
    OP = "=="

class Ne(Comparison):
    OP = "!="

class LtI(Comparison):
    OP = "<"

class LeI(Comparison):
    OP = "<="

class GtI(Comparison):
    OP = ">"

class GeI(Comparison):
    OP = ">="

class LtU(Comparison):
    OP = "<"

class LeU(Comparison):
    OP = "<="

class GtU(Comparison):
    OP = ">"

class GeU(Comparison):
    OP = ">="

class EqF(Comparison):
    OP = "=="

class NeF(Comparison):
    OP = "!="

class LtF(Comparison):
    OP = "<"

class LeF(Comparison):
    OP = "<="

class GtF(Comparison):
    OP = ">"

class GeF(Comparison):
    OP = ">="        

class Deref(Expr):
    def __init__(self, size, target):
        self.size = size
        self.children = [target]
    
    @property
    def target(self):
        return self.children[0]

    def __str__(self):
        return f"*{stringify_expr(self.target)}"
    
    def __repr__(self):
        return f"Deref(size={self.size!r}, target={self.target!r})"

class Ref(Expr):
    def __init__(self, target):
        self.children = [target]

    @property
    def target(self):
        return self.children[0]

    def __str__(self):
        return f"&{stringify_expr(self.target)}"

    def __repr__(self):
        return f"Ref(target={self.target!r})"

class FunCall(Expr):
    def __init__(self, target, args):
        self.children = [target] + args
    
    @property
    def target(self):
        return self.children[0]

    @property
    def args(self):
        return self.children[1:]

    def __str__(self):
        return f"{stringify_expr(self.target)}({', '.join(str(arg) for arg in self.args)})"

    def __repr__(self):
        return f"FunCall(target={self.target!r}, args={self.args!r})"

class BlockCopy(Assign): # TODO: See below.
    """This is sketch? Maybe BlockCopy shouldn't exist and Assign should just take size cause it has multiple stores anyway...
    Counter point:
    typedef struct {                0x00000000  vmMain:             ENTER       0x10                            
        int a;                      0x00000001                      LOCAL       var_8                           
    } s;                            0x00000002                      LOCAL       var_c                           
    void test( void ) {             0x00000003                      BLOCK_COPY  0x4                             
        s x;                        0x00000004                      PUSH                                        
        s y;                        0x00000005                      LEAVE       0x10                            
        x = y;
    }
    Counter counter point: Maybe assign can have a source attribute indicating it originated from BLOCK_COPY or something."""
    def __init__(self, size, lhs, rhs):
        super().__init__(lhs, rhs)
        self.size = size

    def __str__(self):
        return f"{super().__str__()} /* size: {self.size} */"
    
    def __repr__(self):
        return f"BlockCopy(size={self.size}, lhs={self.lhs}, rhs={self.rhs})"

class StackVar(ASTNode):  # TODO: should this derive from an expr or something?
    def __init__(self, offset):
        self.offset = offset
    
    def __str__(self):
        return f"local_{self.offset:x}"
    
    def __repr__(self):
        return f"StackVar(offset={self.offset:#x})"

def stringify_expr(expr): # TODO: Maybe a pretty printer visitor pass or something should be the one to handle these kinds of things...
    if isinstance(expr, Constant) or isinstance(expr, StackVar):
        return str(expr)
    return f"({expr})"

class Stmt(ASTNode):
    pass

class ExprStmt(Stmt):
    def __init__(self, expr):
        self.children[0] = expr
    
    @property
    def expr(self):
        return self.children[0]
    
    def __str__(self):
        return f"{self.expr};"
    
    def __repr__(self):
        return f"ExprStmt(expr={self.expr!r})"

class Block(Stmt):
    def __init__(self, stmts):
        self.children = stmts
    
    def __str__(self): # TODO: implicit vs explicit thing
        lf_tab = "\n\t"
        return f"{{\n\t{lf_tab.join(str(child) for child in self.children)}}}"
    
    def __repr__(self):
        return f"Block(stmts={self.children!r})"

class Goto(Stmt):
    def __init__(self, target):
        self.children = [target]
    
    @property
    def target(self):
        return self.children[0]
    
    def __str__(self):
        return f"goto {self.target};"
    
    def __repr__(self):
        return f"Goto(target={self.target!r})"

class Return(Stmt):
    def __init__(self, value):
        self.children = [value]
    
    @property
    def value(self):
        return self.children[0]
    
    def __str__(self):
        if self.value is None:
            return "return;"
        return f"return {self.value};"
    
    def __repr__(self):
        return f"Return(value={self.value!r})"
    

def _read_int_32(f):
    return int.from_bytes(f.read(4), "little", signed=True)

comparison_map = {
    Opcode.EQ: Eq,
    Opcode.NE: Ne,
    Opcode.LTI: LtI,
    Opcode.LEI: LeI,
    Opcode.GTI: GtI,
    Opcode.GEI: GeI,
    Opcode.LTU: LtU,
    Opcode.LEU: LeU,
    Opcode.GTU: GtU,
    Opcode.GEU: GeU,
    Opcode.EQF: EqF,
    Opcode.NEF: NeF,
    Opcode.LTF: LtF,
    Opcode.LEF: LeF,
    Opcode.GTF: GtF,
    Opcode.GEF: GeF
}

bin_op_map = {
    Opcode.ADD: Add,
    Opcode.SUB: Sub,
    Opcode.DIVI: DivI,
    Opcode.DIVU: DivU,
    Opcode.MODI: ModI,
    Opcode.MODU: ModU,
    Opcode.MULI: MulI,
    Opcode.MULU: MulU,
    Opcode.BAND: And,
    Opcode.BOR: Or,
    Opcode.BXOR: Xor,
    Opcode.LSH: Lsh,
    Opcode.RSHI: RshI,
    Opcode.RSHU: RshU,
    Opcode.ADDF: AddF,
    Opcode.SUBF: SubF,
    Opcode.DIVF: DivF,
    Opcode.MULF: MulF
}

unary_op_map = {
    Opcode.NEGI: NegI,
    Opcode.NEGF: NegF,
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
    for i in range(instruction_count):
        match code.read(1)[0]:
            case Opcode.UNDEF | Opcode.IGNORE | Opcode.BREAK:
                pass
            case Opcode.ENTER:
                assert len(op_stack) == 0, (hex(i), op_stack)
                if ast_nodes:
                    print("\n".join(str(n) for n in ast_nodes))
                ast_nodes = []
                stack_size = _read_int_32(code)
            case Opcode.LEAVE:
                ast_nodes.append(Return(op_stack.pop()))
                stack_size = _read_int_32(code) # TODO: assert mb
            case Opcode.CALL:
                op_stack.append(FunCall(op_stack.pop(), call_args))
                call_args = []
            case Opcode.PUSH:
                op_stack.append(None) # TODO: hmmmm
            case Opcode.POP:
                ast_nodes.append(op_stack.pop())
                assert len(op_stack) == 0, (hex(i), op_stack) # TODO: is this right?
            case Opcode.CONST:
                op_stack.append(Constant(_read_int_32(code)))
            case Opcode.LOCAL:
                offset = _read_int_32(code)
                op_stack.append(Ref(StackVar(offset))) # TODO: what about args?
            case Opcode.JUMP:
                ast_nodes.append(Goto(op_stack.pop()))
                 # TODO: make basic block or store info needed to make one
            case op if op in comparison_map:
                target = _read_int_32(code)
                rhs, lhs = op_stack.pop(), op_stack.pop()
                ast_nodes.append(comparison_map[op](lhs, rhs))
                 # TODO: make basic block or store info needed to make one
            case op if op in bin_op_map:
                rhs, lhs = op_stack.pop(), op_stack.pop()
                op_stack.append(bin_op_map[op](lhs, rhs))
            case (Opcode.LOAD1 | Opcode.LOAD2 | Opcode.LOAD4) as op:
                size = int(2**(op - Opcode.LOAD1)) # TODO: unsketch
                op_stack.append(Deref(size, op_stack.pop()))
            case (Opcode.STORE1 | Opcode.STORE2 | Opcode.STORE4) as op:
                rhs, lhs = op_stack.pop(), op_stack.pop()
                size = int(2**(op - Opcode.STORE1)) # TODO: unsketch
                ast_nodes.append(Assign(Deref(size, lhs), rhs))
            case Opcode.ARG:
                arg = op_stack.pop()
                todo = code.read(1)[0] # TODO: use this...
                call_args.append(arg)
            case Opcode.BLOCK_COPY:
                size = _read_int_32(code)
                rhs, lhs = op_stack.pop(), op_stack.pop()
                ast_nodes.append(BlockCopy(size, lhs, rhs))
            case Opcode.SEX8:
                pass
            case Opcode.SEX16:
                pass
            case op if op in unary_op_map:
                op_stack.append(unary_op_map[op](op_stack.pop()))
            case Opcode.CVIF:
                pass
            case Opcode.CVFI:
                pass
            case op:
                raise Exception(f"Invalid opcode: {op}") # TODO: Find a better class to throw?
    print("\n".join(str(n) for n in ast_nodes))