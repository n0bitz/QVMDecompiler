from enum import IntEnum


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
    Opcode.BLOCK_COPY,
}


def _read_int_32(f):
    return int.from_bytes(f.read(4), "little", signed=True)


def _read_int_8(f):
    return f.read(1)[0]


class Instruction:
    def __init__(self, addr, op, arg=None):
        self.addr = addr
        self.op = Opcode(op)
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
                self.instructions.append(Instruction(i, opcode, arg))
                if opcode == Opcode.ENTER:
                    self.func_addrs.append(i)

    def get_function(self, addr):
        assert self.instructions[addr].op == Opcode.ENTER
        end_addr = addr + 1
        while end_addr < len(self.instructions) and self.instructions[end_addr].op != Opcode.ENTER:
            end_addr += 1
        return self.instructions[addr:end_addr]
