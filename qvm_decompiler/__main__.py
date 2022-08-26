from abc import ABC, abstractmethod
from ast import AST
from sys import argv
from .qvm import Qvm, Instruction
from .opcode import Opcode


class ASTNode(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):
        return type(other) == type(self) and [i for i in self] == [i for i in other]


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
        yield from super().__iter__()
        yield "size", self.size

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
        return self.visit(self.tree), self.replace_count

    def visit_ASTNode(self, node):
        if node == self.old:
            self.replace_count += 1
            return self.new
        self.generic_visit(node)
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


class BasicBlock:
    def __init__(self, nodes, id):
        self.nodes = nodes
        self.predecessors = []
        self.successors = []
        self.id = id


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
                assert len(op_stack) == 0, (hex(i), op_stack)  # TODO: is this right?
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


qvm = Qvm(argv[1])
for func_addr in qvm.func_addrs:
    basic_block_leaders = set()
    # map from the branching instruction index to its successors' indices
    branch_successors_map = {}
    instructions = qvm.get_function(func_addr)

    def add_leader(branch_idx, leader_addr, conditional):
        if branch_idx not in branch_successors_map:
            branch_successors_map[branch_idx] = []
        next_inst_addr = branch_idx + 1
        if next_inst_addr < len(instructions):
            if conditional:
                branch_successors_map[branch_idx].append(next_inst_addr)
            basic_block_leaders.add(next_inst_addr)
        leader_idx = leader_addr - func_addr
        branch_successors_map[branch_idx].append(leader_idx)
        basic_block_leaders.add(leader_idx)

    for i, inst in enumerate(instructions):
        match inst.op:
            case Opcode.JUMP:
                if instructions[i - 1].op == Opcode.CONST:
                    add_leader(i, instructions[i - 1].arg, False)
                else:
                    min_bound = max_bound = switch_base = None
                    match instructions[:i]:  # TODO: change the LSH 2 to actually check for 2 only
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
                            Instruction(op=Opcode.LOAD4),
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
                            Instruction(op=Opcode.LOAD4),
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
                            Instruction(op=Opcode.LOAD4),
                        ]:
                            min_bound, max_bound = minb, maxb
                            switch_base = base
                    if None in (min_bound, max_bound, switch_base):
                        raise Exception(f"Unrecognized switch-case @ {i:#x}")
                    num_entries = max_bound - min_bound + 1
                    switch_base += min_bound * 4
                    for addr in range(switch_base, switch_base + num_entries * 4, 4):
                        add_leader(
                            i,
                            int.from_bytes(qvm.data[addr : addr + 4], "little"),
                            False,
                        )
            case op if op in comparison_map:
                add_leader(i, inst.arg, True)

    basic_blocks = {}
    # map from the block id (index of first instruction in block) to its successor block ids
    block_successors_map = {}
    block_start_idx = 0
    for block_end_idx in sorted(basic_block_leaders | {len(instructions)}):
        if len(block_instructions := instructions[block_start_idx:block_end_idx]):
            if (branch_idx := block_end_idx - 1) in branch_successors_map:
                block_successors_map[block_start_idx] = branch_successors_map[branch_idx]
            block_nodes = astify(block_instructions)
            if len(block_nodes) and isinstance(block_nodes[-1], Goto):
                block_nodes = block_nodes[:-1]
            basic_blocks[block_start_idx] = BasicBlock(block_nodes, block_start_idx + func_addr)
            block_start_idx = block_end_idx

    for block_start_idx, successors in block_successors_map.items():
        curr_block = basic_blocks[block_start_idx]
        for succ_block_idx in successors:
            succ_block = basic_blocks[succ_block_idx]
            curr_block.successors.append(succ_block)
            succ_block.predecessors.append(curr_block)

    basic_blocks = sorted(basic_blocks.values(), key=lambda b: b.id)
    for i, block in enumerate(basic_blocks):
        if len(block.successors) == 0 and i + 1 < len(basic_blocks):
            next_block = basic_blocks[i + 1]
            block.successors.append(next_block)
            next_block.predecessors.append(block)

    deref_ref_simplifier = DerefRefSimplifier()
    for block in basic_blocks:
        nodes = block.nodes
        for i, node in enumerate(nodes):
            nodes[i] = deref_ref_simplifier.visit(node)
        for i in range(len(nodes) - 2, -1, -1):  # TODO: Maybe remove if/when we do data flow analysis?
            match nodes[i]:
                case Assign(lhs=StackVar() as var, rhs=Call() as call):
                    nodes[i + 1], replace_count = NodeReplacer(nodes[i + 1]).replace(var, call)
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
