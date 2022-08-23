from abc import ABC, abstractmethod
from sys import argv
from .qvm import Qvm, Instruction
from .opcode import Opcode

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
    pass

class BCom(UnaryOp):
    pass

class BinOp(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __items__(self):
        yield self.lhs
        yield self.rhs
    
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

class Call(Expr):
    def __init__(self, target, args):
        self.target = target
        self.args = args
    
    def __items__(self):
        yield self.target
        yield from self.args

    def __repr__(self):
        return f"Call(target={self.target!r}, args={self.args!r})"

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

class BasicBlock:
    def __init__(self, nodes):
        self.nodes = nodes
        self.predecessors = []
        self.successors = []

def astify(instructions):
    op_stack = []
    ast_nodes = []
    call_args = []
    for instruction in instructions:
        match instruction.op:
            case Opcode.UNDEF | Opcode.IGNORE | Opcode.BREAK:
                pass
            case Opcode.ENTER:
                stack_size = instruction.arg # TODO: store somewhere or something?
            case Opcode.LEAVE:
                ast_nodes.append(Return(op_stack.pop()))
                stack_size = instruction.arg # TODO: same question as ENTER
            case Opcode.CALL:
                op_stack.append(Call(op_stack.pop(), call_args))
                call_args = []
            case Opcode.PUSH:
                op_stack.append(None)
            case Opcode.POP:
                ast_nodes.append(op_stack.pop())
                assert len(op_stack) == 0, (hex(i), op_stack) # TODO: is this right?
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
                size = int(2**(op - Opcode.LOAD1))
                op_stack.append(Deref(size, op_stack.pop()))
            case (Opcode.STORE1 | Opcode.STORE2 | Opcode.STORE4) as op:
                rhs, lhs = op_stack.pop(), op_stack.pop()
                size = int(2**(op - Opcode.STORE1))
                ast_nodes.append(Assign(Deref(size, lhs), rhs))
            case Opcode.ARG:
                arg = op_stack.pop()
                offset = instruction.arg # TODO: maybe use this one day if needed...
                call_args.append(arg)
            case Opcode.BLOCK_COPY:
                size = instruction.arg
                rhs, lhs = op_stack.pop(), op_stack.pop()
                ast_nodes.append(Assign(Deref(size, lhs), Deref(size, rhs)))
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
                raise Exception(f"Invalid opcode: {op}")
    return ast_nodes

qvm = Qvm(argv[1])
for func_addr in qvm.func_addrs:
    basic_block_leaders = set()
    branch_successors_map = {} # map from the branching instruction index to its successors' indices
    instructions = qvm.get_function(func_addr)
    def add_leader(branch_idx, leader_addr, conditional):
        if branch_idx not in branch_successors_map:
            branch_successors_map[branch_idx] = []
        leader_idx = leader_addr - func_addr
        branch_successors_map[branch_idx].append(leader_idx)
        basic_block_leaders.add(leader_idx)
        next_inst_addr = branch_idx + 1
        if next_inst_addr < len(instructions):
            if conditional:
                branch_successors_map[branch_idx].append(next_inst_addr)
            basic_block_leaders.add(next_inst_addr)
    for i, inst in enumerate(instructions):
        match inst.op:
            case Opcode.JUMP:
                if instructions[i - 1].op == Opcode.CONST:
                    add_leader(i, instructions[i - 1].arg, False)
                else:
                    min_bound = max_bound = switch_base = None
                    match instructions[:i]: # TODO: change the LSH 2 to actually check for 2 only
                        case [
                            *_,
                            Instruction(op=Opcode.LOCAL, arg=temp_offset1),
                            Instruction(op=Opcode.CONST, arg=minb),
                            Instruction(op=Opcode.STORE4),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.LOCAL, arg=temp_offset2),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.LTI),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.CONST, arg=maxb),
                            Instruction(op=Opcode.GTI),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.LSH),
                            Instruction(op=Opcode.CONST, arg=base),
                            Instruction(op=Opcode.ADD),
                            Instruction(op=Opcode.LOAD4)
                        ] if temp_offset1 == temp_offset2:
                            min_bound, max_bound = minb, maxb
                            switch_base = base
                        case [
                            *_,
                            Instruction(op=Opcode.LOCAL, arg=temp_offset1),
                            Instruction(op=Opcode.CONST, arg=maxb),
                            Instruction(op=Opcode.STORE4),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.CONST, arg=minb),
                            Instruction(op=Opcode.LTI),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.LOCAL, arg=temp_offset2),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.GTI),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.LSH),
                            Instruction(op=Opcode.CONST, arg=base),
                            Instruction(op=Opcode.ADD),
                            Instruction(op=Opcode.LOAD4)
                        ] if temp_offset1 == temp_offset2:
                            min_bound, max_bound = minb, maxb
                            switch_base = base
                        case [
                            *_,
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.CONST, arg=minb),
                            Instruction(op=Opcode.LTI),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.CONST, arg=maxb),
                            Instruction(op=Opcode.GTI),
                            Instruction(op=Opcode.LOCAL),
                            Instruction(op=Opcode.LOAD4),
                            Instruction(op=Opcode.CONST),
                            Instruction(op=Opcode.LSH),
                            Instruction(op=Opcode.CONST, arg=base),
                            Instruction(op=Opcode.ADD),
                            Instruction(op=Opcode.LOAD4)
                        ]:
                            min_bound, max_bound = minb, maxb
                            switch_base = base
                    if None in (min_bound, max_bound, switch_base):
                        raise Exception(f"Unrecognized switch-case @ {i:#x}")
                    num_entries = max_bound - min_bound + 1
                    switch_base += min_bound * 4
                    for addr in range(switch_base, switch_base + num_entries * 4, 4):
                        add_leader(i, int.from_bytes(qvm.data[addr:addr + 4], 'little'), False)
            case op if op in comparison_map:
                add_leader(i, inst.arg, True)
    
    basic_blocks = {}
    block_successors_map = {} # map from the block id (index of first instruction in block) to its successor block ids
    block_start_idx = 0
    for block_end_idx in sorted(basic_block_leaders | {len(instructions)}):
        if len(block_instructions := instructions[block_start_idx:block_end_idx]):
            if (branch_idx := block_end_idx - 1) in branch_successors_map:
                block_successors_map[block_start_idx] = branch_successors_map[branch_idx]
            basic_blocks[block_start_idx] = BasicBlock(astify(block_instructions))
            block_start_idx = block_end_idx
            
    for block_start_idx, successors in block_successors_map.items():
        curr_block = basic_blocks[block_start_idx]
        for succ_block_idx in successors:
            succ_block = basic_blocks[succ_block_idx]
            curr_block.successors.append(succ_block)
            succ_block.predecessors.append(curr_block)
    
    print(f"sub_{func_addr:x}:")
    for key in sorted(basic_blocks.keys()):
        print(f"block_{key:x}:")
        block = basic_blocks[key]
        for node in block.nodes:
            print(node)
        print("successors:", [f"block_{list(basic_blocks.keys())[list(basic_blocks.values()).index(i)]:x}" for i in block.successors])
        print()
