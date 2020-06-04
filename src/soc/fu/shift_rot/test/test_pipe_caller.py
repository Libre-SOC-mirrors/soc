from nmigen import Module, Signal
from nmigen.back.pysim import Simulator, Delay, Settle
from nmutil.formaltest import FHDLTestCase
from nmigen.cli import rtlil
import unittest
from soc.decoder.isa.caller import ISACaller, special_sprs
from soc.decoder.power_decoder import (create_pdecode)
from soc.decoder.power_decoder2 import (PowerDecode2)
from soc.decoder.power_enums import (XER_bits, Function, CryIn)
from soc.decoder.selectable_int import SelectableInt
from soc.simulator.program import Program
from soc.decoder.isa.all import ISA


from soc.fu.test.common import TestCase
from soc.fu.shift_rot.pipeline import ShiftRotBasePipe
from soc.fu.alu.alu_input_record import CompALUOpSubset
from soc.fu.shift_rot.pipe_data import ShiftRotPipeSpec
import random


def get_cu_inputs(dec2, sim):
    """naming (res) must conform to ShiftRotFunctionUnit input regspec
    """
    res = {}

    # RA
    reg1_ok = yield dec2.e.read_reg1.ok
    if reg1_ok:
        data1 = yield dec2.e.read_reg1.data
        res['ra'] = sim.gpr(data1).value

    # RB
    reg2_ok = yield dec2.e.read_reg2.ok
    if reg2_ok:
        data2 = yield dec2.e.read_reg2.data
        res['rb'] = sim.gpr(data2).value

    # RS (RC)
    reg3_ok = yield dec2.e.read_reg3.ok
    if reg3_ok:
        data3 = yield dec2.e.read_reg3.data
        res['rc'] = sim.gpr(data3).value

    # XER.ca
    cry_in = yield dec2.e.input_carry
    if cry_in == CryIn.CA.value:
        carry = 1 if sim.spr['XER'][XER_bits['CA']] else 0
        carry32 = 1 if sim.spr['XER'][XER_bits['CA32']] else 0
        res['xer_ca'] = carry | (carry32<<1)

    print ("inputs", res)

    return res


def set_alu_inputs(alu, dec2, sim):
    # TODO: see https://bugs.libre-soc.org/show_bug.cgi?id=305#c43
    # detect the immediate here (with m.If(self.i.ctx.op.imm_data.imm_ok))
    # and place it into data_i.b

    inp = yield from get_cu_inputs(dec2, sim)
    if 'ra' in inp:
        yield alu.p.data_i.a.eq(inp['ra'])
    else:
        yield alu.p.data_i.a.eq(0)
    if 'rb' in inp:
        yield alu.p.data_i.rb.eq(inp['rb'])
    else:
        yield alu.p.data_i.rb.eq(0)
    if 'rc' in inp:
        yield alu.p.data_i.rs.eq(inp['rc'])
    else:
        yield alu.p.data_i.rs.eq(0)

    # If there's an immediate, set the B operand to that
    imm_ok = yield dec2.e.imm_data.imm_ok
    if imm_ok:
        data2 = yield dec2.e.imm_data.imm
        yield alu.p.data_i.rb.eq(data2)

    if 'xer_ca' in inp:
        yield alu.p.data_i.xer_ca.eq(inp['xer_ca'])
        print ("extra inputs: CA/32", bin(inp['xer_ca']))
    else:
        yield alu.p.data_i.xer_ca.eq(0)


# This test bench is a bit different than is usual. Initially when I
# was writing it, I had all of the tests call a function to create a
# device under test and simulator, initialize the dut, run the
# simulation for ~2 cycles, and assert that the dut output what it
# should have. However, this was really slow, since it needed to
# create and tear down the dut and simulator for every test case.

# Now, instead of doing that, every test case in ShiftRotTestCase puts some
# data into the test_data list below, describing the instructions to
# be tested and the initial state. Once all the tests have been run,
# test_data gets passed to TestRunner which then sets up the DUT and
# simulator once, runs all the data through it, and asserts that the
# results match the pseudocode sim at every cycle.

# By doing this, I've reduced the time it takes to run the test suite
# massively. Before, it took around 1 minute on my computer, now it
# takes around 3 seconds


class ShiftRotTestCase(FHDLTestCase):
    test_data = []
    def __init__(self, name):
        super().__init__(name)
        self.test_name = name

    def run_tst_program(self, prog, initial_regs=None, initial_sprs=None):
        tc = TestCase(prog, self.test_name, initial_regs, initial_sprs)
        self.test_data.append(tc)

    def test_shift(self):
        insns = ["slw", "sld", "srw", "srd", "sraw", "srad"]
        for i in range(20):
            choice = random.choice(insns)
            lst = [f"{choice} 3, 1, 2"]
            initial_regs = [0] * 32
            initial_regs[1] = random.randint(0, (1<<64)-1)
            initial_regs[2] = random.randint(0, 63)
            print(initial_regs[1], initial_regs[2])
            self.run_tst_program(Program(lst), initial_regs)

    def test_shift_arith(self):
        lst = ["sraw 3, 1, 2"]
        initial_regs = [0] * 32
        initial_regs[1] = random.randint(0, (1<<64)-1)
        initial_regs[2] = random.randint(0, 63)
        print(initial_regs[1], initial_regs[2])
        self.run_tst_program(Program(lst), initial_regs)

    def test_shift_once(self):
        lst = ["slw 3, 1, 4",
               "slw 3, 1, 2"]
        initial_regs = [0] * 32
        initial_regs[1] = 0x80000000
        initial_regs[2] = 0x40
        initial_regs[4] = 0x00
        self.run_tst_program(Program(lst), initial_regs)

    def test_rlwinm(self):
        for i in range(10):
            mb = random.randint(0,31)
            me = random.randint(0,31)
            sh = random.randint(0,31)
            lst = [f"rlwinm 3, 1, {mb}, {me}, {sh}",
                   #f"rlwinm. 3, 1, {mb}, {me}, {sh}"
                   ]
            initial_regs = [0] * 32
            initial_regs[1] = random.randint(0, (1<<64)-1)
            self.run_tst_program(Program(lst), initial_regs)

    def test_rlwimi(self):
        lst = ["rlwimi 3, 1, 5, 20, 6"]
        initial_regs = [0] * 32
        initial_regs[1] = 0xdeadbeef
        initial_regs[3] = 0x12345678
        self.run_tst_program(Program(lst), initial_regs)

    def test_rlwnm(self):
        lst = ["rlwnm 3, 1, 2, 20, 6"]
        initial_regs = [0] * 32
        initial_regs[1] = random.randint(0, (1<<64)-1)
        initial_regs[2] = random.randint(0, 63)
        self.run_tst_program(Program(lst), initial_regs)

    def test_rldicl(self):
        lst = ["rldicl 3, 1, 5, 20"]
        initial_regs = [0] * 32
        initial_regs[1] = random.randint(0, (1<<64)-1)
        self.run_tst_program(Program(lst), initial_regs)

    def test_rldicr(self):
        lst = ["rldicr 3, 1, 5, 20"]
        initial_regs = [0] * 32
        initial_regs[1] = random.randint(0, (1<<64)-1)
        self.run_tst_program(Program(lst), initial_regs)

    def test_rlc(self):
        insns = ["rldic", "rldicl", "rldicr"]
        for i in range(20):
            choice = random.choice(insns)
            sh = random.randint(0, 63)
            m = random.randint(0, 63)
            lst = [f"{choice} 3, 1, {sh}, {m}"]
            initial_regs = [0] * 32
            initial_regs[1] = random.randint(0, (1<<64)-1)
            self.run_tst_program(Program(lst), initial_regs)

    def test_ilang(self):
        pspec = ShiftRotPipeSpec(id_wid=2)
        alu = ShiftRotBasePipe(pspec)
        vl = rtlil.convert(alu, ports=alu.ports())
        with open("pipeline.il", "w") as f:
            f.write(vl)


class TestRunner(FHDLTestCase):
    def __init__(self, test_data):
        super().__init__("run_all")
        self.test_data = test_data

    def run_all(self):
        m = Module()
        comb = m.d.comb
        instruction = Signal(32)

        pdecode = create_pdecode()

        m.submodules.pdecode2 = pdecode2 = PowerDecode2(pdecode)

        pspec = ShiftRotPipeSpec(id_wid=2)
        m.submodules.alu = alu = ShiftRotBasePipe(pspec)

        comb += alu.p.data_i.ctx.op.eq_from_execute1(pdecode2.e)
        comb += alu.p.valid_i.eq(1)
        comb += alu.n.ready_i.eq(1)
        comb += pdecode2.dec.raw_opcode_in.eq(instruction)
        sim = Simulator(m)

        sim.add_clock(1e-6)
        def process():
            for test in self.test_data:
                print(test.name)
                program = test.program
                self.subTest(test.name)
                simulator = ISA(pdecode2, test.regs, test.sprs, 0)
                gen = program.generate_instructions()
                instructions = list(zip(gen, program.assembly.splitlines()))

                index = simulator.pc.CIA.value//4
                while index < len(instructions):
                    ins, code = instructions[index]

                    print("0x{:X}".format(ins & 0xffffffff))
                    print(code)

                    # ask the decoder to decode this binary data (endian'd)
                    yield pdecode2.dec.bigendian.eq(0)  # little / big?
                    yield instruction.eq(ins)          # raw binary instr.
                    yield Settle()
                    fn_unit = yield pdecode2.e.fn_unit
                    self.assertEqual(fn_unit, Function.SHIFT_ROT.value)
                    yield from set_alu_inputs(alu, pdecode2, simulator)
                    yield
                    opname = code.split(' ')[0]
                    yield from simulator.call(opname)
                    index = simulator.pc.CIA.value//4

                    vld = yield alu.n.valid_o
                    while not vld:
                        yield
                        vld = yield alu.n.valid_o
                    yield
                    alu_out = yield alu.n.data_o.o.data
                    out_reg_valid = yield pdecode2.e.write_reg.ok
                    if out_reg_valid:
                        write_reg_idx = yield pdecode2.e.write_reg.data
                        expected = simulator.gpr(write_reg_idx).value
                        msg = f"expected {expected:x}, actual: {alu_out:x}"
                        self.assertEqual(expected, alu_out, msg)
                    yield from self.check_extra_alu_outputs(alu, pdecode2,
                                                            simulator)
                    break

        sim.add_sync_process(process)
        with sim.write_vcd("simulator.vcd", "simulator.gtkw",
                            traces=[]):
            sim.run()

    def check_extra_alu_outputs(self, alu, dec2, sim):
        rc = yield dec2.e.rc.data
        if rc:
            cr_expected = sim.crl[0].get_range().value
            cr_actual = yield alu.n.data_o.cr0
            self.assertEqual(cr_expected, cr_actual)


if __name__ == "__main__":
    unittest.main(exit=False)
    suite = unittest.TestSuite()
    suite.addTest(TestRunner(ShiftRotTestCase.test_data))

    runner = unittest.TextTestRunner()
    runner.run(suite)
