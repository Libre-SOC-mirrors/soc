from nmigen import Module, Signal
from nmigen.back.pysim import Simulator, Delay
from nmigen.test.utils import FHDLTestCase
import unittest
from soc.decoder.power_decoder import (create_pdecode)
from soc.decoder.power_enums import (Function, InternalOp,
                                     In1Sel, In2Sel, In3Sel,
                                     OutSel, RC, LdstLen, CryIn,
                                     single_bit_flags, Form,
                                     get_signal_name, get_csv)
from soc.decoder.power_decoder2 import (PowerDecode2)
import tempfile
import subprocess
import struct
import random
import pdb



class Register:
    def __init__(self, num):
        self.num = num

class RegRegOp:
    def __init__(self):
        self.ops = {
            "add": InternalOp.OP_ADD,
            "and": InternalOp.OP_AND,
            "or": InternalOp.OP_OR,
            "add.": InternalOp.OP_ADD,
            "lwzx": InternalOp.OP_LOAD,
            "stwx": InternalOp.OP_STORE,
        }
        self.opcodestr = random.choice(list(self.ops.keys()))
        self.opcode = self.ops[self.opcodestr]
        self.r1 = Register(random.randrange(32))
        self.r2 = Register(random.randrange(32))
        self.r3 = Register(random.randrange(32))

    def generate_instruction(self):
        string = "{} {}, {}, {}\n".format(self.opcodestr,
                                          self.r1.num,
                                          self.r2.num,
                                          self.r3.num)
        return string

    def check_results(self, pdecode2):
        if self.opcode == InternalOp.OP_STORE:
            r1sel = yield pdecode2.e.read_reg3.data
        else:
            r1sel = yield pdecode2.e.write_reg.data

        r3sel = yield pdecode2.e.read_reg2.data

        # For some reason r2 gets decoded either in read_reg1
        # or read_reg3
        out_sel = yield pdecode2.dec.op.out_sel
        if out_sel == OutSel.RA.value:
            r2sel = yield pdecode2.e.read_reg3.data
        else:
            r2sel = yield pdecode2.e.read_reg1.data
        assert(r1sel == self.r1.num)
        assert(r3sel == self.r3.num)
        assert(r2sel == self.r2.num)

        opc_out = yield pdecode2.dec.op.internal_op
        assert(opc_out == self.opcode.value)
        # check RC value (the dot in the instruction)
        rc = yield pdecode2.e.rc.data
        if '.' in self.opcodestr:
            assert(rc == 1)
        else:
            assert(rc == 0)



class RegImmOp:
    def __init__(self):
        self.ops = {
            "addi": InternalOp.OP_ADD,
            "addis": InternalOp.OP_ADD,
            "andi.": InternalOp.OP_AND,
            "ori": InternalOp.OP_OR,
        }
        self.opcodestr = random.choice(list(self.ops.keys()))
        self.opcode = self.ops[self.opcodestr]
        self.r1 = Register(random.randrange(32))
        self.r2 = Register(random.randrange(32))
        self.imm = random.randrange(32767)

    def generate_instruction(self):
        string = "{} {}, {}, {}\n".format(self.opcodestr,
                                          self.r1.num,
                                          self.r2.num,
                                          self.imm)
        return string

    def check_results(self, pdecode2):
        print("Check")
        r1sel = yield pdecode2.e.write_reg.data
        # For some reason r2 gets decoded either in read_reg1
        # or read_reg3
        out_sel = yield pdecode2.dec.op.out_sel
        if out_sel == OutSel.RA.value:
            r2sel = yield pdecode2.e.read_reg3.data
        else:
            r2sel = yield pdecode2.e.read_reg1.data
        assert(r1sel == self.r1.num)
        assert(r2sel == self.r2.num)

        imm = yield pdecode2.e.imm_data.data
        in2_sel = yield pdecode2.dec.op.in2_sel
        if in2_sel in [In2Sel.CONST_SI_HI.value, In2Sel.CONST_UI_HI.value]:
            assert(imm == (self.imm << 16))
        else:
            assert(imm == self.imm)

        rc = yield pdecode2.e.rc.data
        if '.' in self.opcodestr:
            assert(rc == 1)
        else:
            assert(rc == 0)

class LdStOp:
    def __init__(self):
        self.ops = {
            "lwz": InternalOp.OP_LOAD,
            "stw": InternalOp.OP_STORE,
            "lwzu": InternalOp.OP_LOAD,
            "stwu": InternalOp.OP_STORE,
            "lbz": InternalOp.OP_LOAD,
            "lhz": InternalOp.OP_LOAD,
            "stb": InternalOp.OP_STORE,
            "sth": InternalOp.OP_STORE,
        }
        self.opcodestr = random.choice(list(self.ops.keys()))
        self.opcode = self.ops[self.opcodestr]
        self.r1 = Register(random.randrange(32))
        self.r2 = Register(random.randrange(1,32))
        self.imm = random.randrange(32767)

    def generate_instruction(self):
        string = "{} {}, {}({})\n".format(self.opcodestr,
                                          self.r1.num,
                                          self.imm,
                                          self.r2.num)
        return string

    def check_results(self, pdecode2):
        print("Check")
        r2sel = yield pdecode2.e.read_reg1.data
        if self.opcode == InternalOp.OP_STORE:
            r1sel = yield pdecode2.e.read_reg3.data
        else:
            r1sel = yield pdecode2.e.write_reg.data
        assert(r1sel == self.r1.num)
        assert(r2sel == self.r2.num)

        imm = yield pdecode2.e.imm_data.data
        in2_sel = yield pdecode2.dec.op.in2_sel
        assert(imm == self.imm)

        update = yield pdecode2.e.update
        if "u" in self.opcodestr:
            assert(update == 1)
        else:
            assert(update == 0)

        size = yield pdecode2.e.data_len
        if "w" in self.opcodestr:
            assert(size == 4)
        elif "h" in self.opcodestr:
            assert(size == 2)
        elif "b" in self.opcodestr:
            assert(size == 1)
        else:
            assert(False)


class CmpRegOp:
    def __init__(self):
        self.ops = {
            "cmp": InternalOp.OP_CMP,
        }
        self.opcodestr = random.choice(list(self.ops.keys()))
        self.opcode = self.ops[self.opcodestr]
        self.r1 = Register(random.randrange(32))
        self.r2 = Register(random.randrange(32))
        self.cr = Register(random.randrange(8))

    def generate_instruction(self):
        string = "{} {}, 0, {}, {}\n".format(self.opcodestr,
                                             self.cr.num,
                                             self.r1.num,
                                             self.r2.num)
        return string

    def check_results(self, pdecode2):
        r1sel = yield pdecode2.e.read_reg1.data
        r2sel = yield pdecode2.e.read_reg2.data
        crsel = yield pdecode2.dec.BF[0:-1]

        assert(r1sel == self.r1.num)
        assert(r2sel == self.r2.num)
        assert(crsel == self.cr.num)


class RotateOp:
    def __init__(self):
        self.ops = {
            "rlwinm": InternalOp.OP_CMP,
            "rlwnm": InternalOp.OP_CMP,
            "rlwimi": InternalOp.OP_CMP,
            "rlwinm.": InternalOp.OP_CMP,
            "rlwnm.": InternalOp.OP_CMP,
            "rlwimi.": InternalOp.OP_CMP,
        }
        self.opcodestr = random.choice(list(self.ops.keys()))
        self.opcode = self.ops[self.opcodestr]
        self.r1 = Register(random.randrange(32))
        self.r2 = Register(random.randrange(32))
        self.shift = random.randrange(32)
        self.mb = random.randrange(32)
        self.me = random.randrange(32)

    def generate_instruction(self):
        string = "{} {},{},{},{},{}\n".format(self.opcodestr,
                                              self.r1.num,
                                              self.r2.num,
                                              self.shift,
                                              self.mb,
                                              self.me)
        return string

    def check_results(self, pdecode2):
        r1sel = yield pdecode2.e.write_reg.data
        r2sel = yield pdecode2.e.read_reg3.data
        dec = pdecode2.dec

        if "i" in self.opcodestr:
            shift = yield dec.SH[0:-1]
        else:
            shift = yield pdecode2.e.read_reg2.data
        mb = yield dec.MB[0:-1]
        me = yield dec.ME[0:-1]

        assert(r1sel == self.r1.num)
        assert(r2sel == self.r2.num)
        assert(shift == self.shift)
        assert(mb == self.mb)
        assert(me == self.me)

        rc = yield pdecode2.e.rc.data
        if '.' in self.opcodestr:
            assert(rc == 1)
        else:
            assert(rc == 0)


class Branch:
    def __init__(self):
        self.ops = {
            "b": InternalOp.OP_B,
            "ba": InternalOp.OP_B,
            "bla": InternalOp.OP_B,
        }
        self.opcodestr = random.choice(list(self.ops.keys()))
        self.opcode = self.ops[self.opcodestr]
        self.addr = random.randrange(2**23) * 4

    def generate_instruction(self):
        string = "{} {}\n".format(self.opcodestr,
                                  self.addr)
        return string

    def check_results(self, pdecode2):
        imm = yield pdecode2.e.imm_data.data

        assert(imm == self.addr)
        lk = yield pdecode2.e.lk
        if "l" in self.opcodestr:
            assert(lk == 1)
        else:
            assert(lk == 0)
        aa = yield pdecode2.dec.AA[0:-1]
        if "a" in self.opcodestr:
            assert(aa == 1)
        else:
            assert(aa == 0)


class DecoderTestCase(FHDLTestCase):

    def get_assembled_instruction(self, instruction):
        with tempfile.NamedTemporaryFile(suffix=".o") as outfile:
            args = ["powerpc64-linux-gnu-as",
                    "-o",
                    outfile.name]
            p = subprocess.Popen(args, stdin=subprocess.PIPE)
            p.communicate(instruction.encode('utf-8'))
            assert(p.wait() == 0)

            with tempfile.NamedTemporaryFile(suffix=".bin") as binfile:
                args = ["powerpc64-linux-gnu-objcopy",
                        "-O", "binary",
                        outfile.name,
                        binfile.name]
                subprocess.check_output(args)
                binary = struct.unpack('>i', binfile.read(4))[0]
                return binary

    def run_tst(self, kls, name):
        m = Module()
        comb = m.d.comb
        instruction = Signal(32)

        pdecode = create_pdecode()

        m.submodules.pdecode2 = pdecode2 = PowerDecode2(pdecode)
        comb += pdecode2.dec.opcode_in.eq(instruction)

        sim = Simulator(m)

        def process():
            for i in range(20):
                checker = kls()

                instruction_str = checker.generate_instruction()
                print("instr", instruction_str.strip())
                instruction_bin = self.get_assembled_instruction(
                    instruction_str)
                print("code", hex(instruction_bin), bin(instruction_bin))

                yield instruction.eq(instruction_bin)
                yield Delay(1e-6)

                yield from checker.check_results(pdecode2)


        sim.add_process(process)
        with sim.write_vcd("%s.vcd" % name, "%s.gtkw" % name,
                           traces=[pdecode2.ports()]):
            sim.run()
    def test_reg_reg(self):
        self.run_tst(RegRegOp, "reg_reg")

    def test_reg_imm(self):
        self.run_tst(RegImmOp, "reg_imm")

    def test_ldst_imm(self):
        self.run_tst(LdStOp, "ldst_imm")

    def test_cmp_reg(self):
        self.run_tst(CmpRegOp, "cmp_reg")

    def test_rot(self):
        self.run_tst(RotateOp, "rot")

    def test_branch(self):
        self.run_tst(Branch, "branch_abs")

if __name__ == "__main__":
    unittest.main()
