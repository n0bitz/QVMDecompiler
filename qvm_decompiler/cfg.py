from .ast import Goto, astify, comparison_map
from .qvm import Instruction, Opcode


class BasicBlock:
    def __init__(self, nodes, id):
        self.nodes = nodes
        self.predecessors = []
        self.successors = []
        self.id = id


def build_cfg(qvm):
    functions = {}
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

        functions[func_addr] = basic_blocks
    return functions
