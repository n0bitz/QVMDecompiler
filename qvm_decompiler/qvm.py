from .opcode import Opcode

argged_insts = {
    Opcode.ENTER,
    Opcode.LEAVE,
    Opcode.CONST,
    Opcode.LOCAL,
    Opcode.EQ,
    Opcode.NE,
    Opcode.LTI,
    Opcode.LEI,
    Opcode.GTI,
    Opcode.GEI,
    Opcode.LTU,
    Opcode.LEU,
    Opcode.GTU,
    Opcode.GEU,
    Opcode.EQF,
    Opcode.NEF,
    Opcode.LTF,
    Opcode.LEF,
    Opcode.GTF,
    Opcode.GEF,
    Opcode.ARG,
    Opcode.BLOCK_COPY
}

def _read_int_32(f):
    return int.from_bytes(f.read(4), "little", signed=True)

def _read_int_8(f):
    return f.read(1)[0]

class Instruction:
    def __init__(self, op, arg=None):
        self.op = op
        self.arg = arg
    
    def __repr__(self):
        if self.arg is not None:
            return f"Instruction(op={Opcode(self.op).name}, arg={self.arg:#x})"
        else:
            return f"Instruction(op={Opcode(self.op).name})"
    
    def __str__(self):
        op = Opcode(self.op).name
        if self.arg is not None:
            return f"{op} {self.arg:#x}"
        return op

# ⁣😥    😫  😒😣😒
# 😒😒  😒 😒    😲
# 😩 😢 😲 ⁣😤    😠
# 😒  😒😒 😞    😤
# 😭    😖  😒😔😫

class Qvm:
    def __init__(self, qvm_path):
        with open(qvm_path, "rb") as f:
            magic = f.read(4)
            assert magic == b"\x44\x14\x72\x12"
            instruction_count = _read_int_32(f)
            code_offset, code_length = _read_int_32(f), _read_int_32(f)
            data_offset, self.data_length = _read_int_32(f), _read_int_32(f)
            self.lit_length, self.bss_length = _read_int_32(f), _read_int_32(f)

            f.seek(data_offset)
            self.data = f.read(self.data_length + self.lit_length)

            f.seek(code_offset)
            self.func_addrs = []
            self.instructions = []
            for i in range(instruction_count):
                opcode, arg = _read_int_8(f), None
                if opcode in argged_insts:
                   arg = _read_int_8(f) if opcode == Opcode.ARG else _read_int_32(f)
                self.instructions.append(Instruction(opcode, arg))
                if opcode == Opcode.ENTER:
                    self.func_addrs.append(i)

    def get_function(self, addr):
        assert self.instructions[addr].op == Opcode.ENTER
        end_addr = addr + 1
        while end_addr < len(self.instructions) and self.instructions[end_addr].op != Opcode.ENTER:
            end_addr += 1
        return self.instructions[addr:end_addr]