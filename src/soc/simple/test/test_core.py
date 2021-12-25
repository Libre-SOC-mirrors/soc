"""simple core test

related bugs:

 * https://bugs.libre-soc.org/show_bug.cgi?id=363
 * https://bugs.libre-soc.org/show_bug.cgi?id=686
"""

from nmigen import Module, Signal, Cat
from nmigen.back.pysim import Simulator, Delay, Settle
from nmutil.formaltest import FHDLTestCase
from nmigen.cli import rtlil
import unittest
from openpower.test.state import (SimState, teststate_check_regs,
                                  teststate_check_mem)
from soc.simple.test.teststate import HDLState
from openpower.decoder.isa.caller import special_sprs
from openpower.decoder.power_decoder import create_pdecode
from openpower.decoder.power_decoder2 import PowerDecode2
from openpower.decoder.selectable_int import SelectableInt
from openpower.decoder.isa.all import ISA
from openpower.decoder.decode2execute1 import IssuerDecode2ToOperand
from openpower.state import CoreState

# note that using SPRreduced has to be done to match the
# PowerDecoder2 SPR map
from openpower.decoder.power_enums import SPRreduced as SPR
from openpower.decoder.power_enums import spr_dict, Function, XER_bits
from soc.config.test.test_loadstore import TestMemPspec
from openpower.endian import bigendian
from soc.regfile.regfiles import StateRegs

from soc.simple.core import NonProductionCore
from soc.experiment.compalu_multi import find_ok  # hack

from soc.fu.compunits.test.test_compunit import (setup_tst_memory,
                                                 check_sim_memory)

# test with ALU data and Logical data
from soc.fu.alu.test.test_pipe_caller import ALUTestCase
from soc.fu.logical.test.test_pipe_caller import LogicalTestCase
from soc.fu.shift_rot.test.test_pipe_caller import ShiftRotTestCase
from soc.fu.cr.test.test_pipe_caller import CRTestCase
from soc.fu.branch.test.test_pipe_caller import BranchTestCase
from soc.fu.ldst.test.test_pipe_caller import LDSTTestCase
from openpower.test.general.overlap_hazards import (HazardTestCase,
                                                    RandomHazardTestCase)
from openpower.util import spr_to_fast_reg

from openpower.consts import StateRegsEnum

# list of SPRs that are controlled and managed by the MMU
mmu_sprs = ["PRTBL", "PIDR"]
ldst_sprs = ["DAR", "DSISR"]


def set_mmu_spr(name, i, val, core):  # important keep pep8 formatting
    fsm = core.fus.get_fu("mmu0").alu
    yield fsm.mmu.l_in.mtspr.eq(1)
    yield fsm.mmu.l_in.sprn.eq(i)
    yield fsm.mmu.l_in.rs.eq(val)
    yield
    yield fsm.mmu.l_in.mtspr.eq(0)
    while True:
        done = yield fsm.mmu.l_out.done
        if done:
            break
        yield
    yield
    print("mmu_spr %s %d was updated %x" % (name, i, val))


def set_ldst_spr(name, i, val, core):  # important keep pep8 formatting
    ldst = core.fus.get_fu("mmu0").alu.ldst # awkward to get at but it works
    yield ldst.sprval_in.eq(val)
    yield ldst.mmu_set_spr.eq(1)
    if name == 'DAR':
        yield ldst.mmu_set_dar.eq(1)
        yield
        yield ldst.mmu_set_dar.eq(0)
    else:
        yield ldst.mmu_set_dsisr.eq(1)
        yield
        yield ldst.mmu_set_dsisr.eq(0)
    yield ldst.mmu_set_spr.eq(0)
    print("ldst_spr %s %d was updated %x" % (name, i, val))


def setup_regs(pdecode2, core, test):

    # set up INT regfile, "direct" write (bypass rd/write ports)
    intregs = core.regs.int
    for i in range(32):
        if intregs.unary:
            yield intregs.regs[i].reg.eq(test.regs[i])
        else:
            yield intregs.memory._array[i].eq(test.regs[i])
    yield Settle()

    # set up MSR in STATE regfile, "direct" write (bypass rd/write ports)
    stateregs = core.regs.state
    yield stateregs.regs[StateRegsEnum.MSR].reg.eq(test.msr)

    # set up CR regfile, "direct" write across all CRs
    cr = test.cr
    crregs = core.regs.cr
    #cr = int('{:32b}'.format(cr)[::-1], 2)
    print("setup cr reg", hex(cr))
    for i in range(8):
        #j = 7-i
        cri = (cr >> (i * 4)) & 0xf
        #cri = int('{:04b}'.format(cri)[::-1], 2)
        print("setup cr reg", hex(cri), i,
              crregs.regs[i].reg.shape())
        yield crregs.regs[i].reg.eq(cri)

    # set up XER.  "direct" write (bypass rd/write ports)
    xregs = core.regs.xer
    print("setup sprs", test.sprs)
    xer = None
    if 'XER' in test.sprs:
        xer = test.sprs['XER']
    if 1 in test.sprs:
        xer = test.sprs[1]
    if xer is not None:
        if isinstance(xer, int):
            xer = SelectableInt(xer, 64)
        sobit = xer[XER_bits['SO']].value
        yield xregs.regs[xregs.SO].reg.eq(sobit)
        cabit = xer[XER_bits['CA']].value
        ca32bit = xer[XER_bits['CA32']].value
        yield xregs.regs[xregs.CA].reg.eq(Cat(cabit, ca32bit))
        ovbit = xer[XER_bits['OV']].value
        ov32bit = xer[XER_bits['OV32']].value
        yield xregs.regs[xregs.OV].reg.eq(Cat(ovbit, ov32bit))
        print("setting XER so %d ca %d ca32 %d ov %d ov32 %d" %
              (sobit, cabit, ca32bit, ovbit, ov32bit))
    else:
        yield xregs.regs[xregs.SO].reg.eq(0)
        yield xregs.regs[xregs.OV].reg.eq(0)
        yield xregs.regs[xregs.CA].reg.eq(0)

    # setting both fast and slow SPRs from test data

    fregs = core.regs.fast
    sregs = core.regs.spr
    for sprname, val in test.sprs.items():
        if isinstance(val, SelectableInt):
            val = val.value
        if isinstance(sprname, int):
            sprname = spr_dict[sprname].SPR
        if sprname == 'XER':
            continue
        print ('set spr %s val %x' % (sprname, val))

        fast = spr_to_fast_reg(sprname)

        if fast is None:
            # match behaviour of SPRMap in power_decoder2.py
            for i, x in enumerate(SPR):
                if sprname == x.name:
                    print("setting slow SPR %d (%s/%d) to %x" %
                          (i, sprname, x.value, val))
                    if sprname in mmu_sprs:
                        yield from set_mmu_spr(sprname, x.value, val, core)
                    elif sprname in ldst_sprs:
                        yield from set_ldst_spr(sprname, x.value, val, core)
                    else:
                        yield sregs.memory._array[i].eq(val)
        else:
            print("setting fast reg %d (%s) to %x" %
                  (fast, sprname, val))
            if fregs.unary:
                rval = fregs.int.regs[fast].reg
            else:
                rval = fregs.memory._array[fast]
            yield rval.eq(val)

    # allow changes to settle before reporting on XER
    yield Settle()

    # XER
    so = yield xregs.regs[xregs.SO].reg
    ov = yield xregs.regs[xregs.OV].reg
    ca = yield xregs.regs[xregs.CA].reg
    oe = yield pdecode2.e.do.oe.oe
    oe_ok = yield pdecode2.e.do.oe.oe_ok

    print("before: so/ov-32/ca-32", so, bin(ov), bin(ca))
    print("oe:", oe, oe_ok)


def check_regs(dut, sim, core, test, code):
    # create the two states and compare
    testdic = {'sim': sim, 'hdl': core}
    yield from teststate_check_regs(dut, testdic, test, code)


def check_mem(dut, sim, core, test, code):
    # create the two states and compare mem
    testdic = {'sim': sim, 'hdl': core}
    yield from teststate_check_mem(dut, testdic, test, code)


def wait_for_busy_hi(cu):
    while True:
        busy_o = yield cu.busy_o
        terminate_o = yield cu.core_terminate_o
        if busy_o:
            print("busy/terminate:", busy_o, terminate_o)
            break
        print("!busy", busy_o, terminate_o)
        yield


def set_issue(core, dec2, sim):
    yield core.issue_i.eq(1)
    yield
    yield core.issue_i.eq(0)
    yield from wait_for_busy_hi(core)


def wait_for_busy_clear(cu):
    while True:
        busy_o = yield cu.o.busy_o
        terminate_o = yield cu.o.core_terminate_o
        if not busy_o:
            print("busy/terminate:", busy_o, terminate_o)
            break
        print("busy",)
        yield


class TestRunner(FHDLTestCase):
    def __init__(self, tst_data):
        super().__init__("run_all")
        self.test_data = tst_data

    def run_all(self):
        m = Module()
        comb = m.d.comb
        instruction = Signal(32)

        units = {'alu': 3, 'cr': 1, 'branch': 1, 'trap': 1,
                 'spr': 1,
                 'logical': 1,
                 'mul': 3,
                 'div': 1, 'shiftrot': 1}

        pspec = TestMemPspec(ldst_ifacetype='testpi',
                             imem_ifacetype='',
                             addr_wid=48,
                             mask_wid=8,
                             units=units,
                             allow_overlap=True,
                             reg_wid=64)

        cur_state = CoreState("cur") # current state (MSR/PC/SVSTATE)
        pdecode2 = PowerDecode2(None, state=cur_state,
                                     #opkls=IssuerDecode2ToOperand,
                                     svp64_en=True, # self.svp64_en,
                                     regreduce_en=False, #self.regreduce_en
                                    )

        m.submodules.core = core = NonProductionCore(pspec)
        m.submodules.pdecode2 = pdecode2
        core.pdecode2 = pdecode2
        l0 = core.l0

        comb += pdecode2.dec.raw_opcode_in.eq(instruction)
        comb += pdecode2.dec.bigendian.eq(bigendian)  # little / big?
        comb += core.i.e.eq(pdecode2.e)
        comb += core.i.state.eq(cur_state)
        comb += core.i.raw_insn_i.eq(instruction)
        comb += core.i.bigendian_i.eq(bigendian)

        # set the PC StateRegs read port to always send back the PC
        stateregs = core.regs.state
        pc_regnum = StateRegs.PC
        comb += stateregs.r_ports['cia'].ren.eq(1<<pc_regnum)

        # temporary hack: says "go" immediately for both address gen and ST
        ldst = core.fus.fus['ldst0']
        m.d.comb += ldst.ad.go_i.eq(ldst.ad.rel_o)  # link addr-go to rel
        m.d.comb += ldst.st.go_i.eq(ldst.st.rel_o)  # link store-go to rel

        # nmigen Simulation
        sim = Simulator(m)
        sim.add_clock(1e-6)

        def process():
            yield

            for test in self.test_data:
                print(test.name)
                program = test.program
                with self.subTest(test.name):
                    sim = ISA(pdecode2, test.regs, test.sprs, test.cr,
                              test.mem,
                              test.msr,
                              bigendian=bigendian)
                    gen = program.generate_instructions()
                    instructions = list(zip(gen, program.assembly.splitlines()))

                    yield from setup_tst_memory(l0, test.mem)
                    yield from setup_regs(pdecode2, core, test)

                    index = sim.pc.CIA.value // 4
                    while index < len(instructions):
                        ins, code = instructions[index]

                        print("instruction: 0x{:X}".format(ins & 0xffffffff))
                        print(code)

                        # ask the decoder to decode this binary data (endian'd)
                        yield instruction.eq(ins)          # raw binary instr.
                        yield Settle()

                        print("sim", code)
                        # call simulated operation
                        opname = code.split(' ')[0]
                        yield from sim.call(opname)
                        pc = sim.pc.CIA.value
                        nia = sim.pc.NIA.value
                        index = pc // 4

                        # set the PC to the same simulated value
                        # (core is not able to do this itself, except
                        # for branch / TRAP)
                        print ("after call, pc nia", pc, nia)
                        yield stateregs.regs[pc_regnum].reg.eq(pc)
                        yield Settle()

                        yield core.p.i_valid.eq(1)
                        yield
                        o_ready = yield core.p.o_ready
                        while True:
                            if o_ready:
                                break
                            yield
                            o_ready = yield core.p.o_ready
                        yield core.p.i_valid.eq(0)

                        # set operand and get inputs
                        yield from wait_for_busy_clear(core)

                        # synchronised (non-overlap) is fine to check
                        if not core.allow_overlap:
                            # register check
                            yield from check_regs(self, sim, core, test, code)

                            # Memory check
                            yield from check_mem(self, sim, core, test, code)

                    # non-overlap mode is only fine to check right at the end
                    if core.allow_overlap:
                        # wait until all settled
                        # XXX really this should be in DMI, which should in turn
                        # use issuer.any_busy to not send back "stopped" signal
                        while (yield core.o.any_busy_o):
                            yield
                        yield Settle()

                        # register check
                        yield from check_regs(self, sim, core, test, code)

                        # Memory check
                        yield from check_mem(self, sim, core, test, code)

            # give a couple extra clock cycles for gtkwave display to be happy
            yield
            yield

        sim.add_sync_process(process)
        with sim.write_vcd("core_simulator.vcd", "core_simulator.gtkw",
                           traces=[]):
            sim.run()


if __name__ == "__main__":
    unittest.main(exit=False)
    suite = unittest.TestSuite()
    suite.addTest(TestRunner(HazardTestCase().test_data))
    suite.addTest(TestRunner(RandomHazardTestCase().test_data))
    #suite.addTest(TestRunner(LDSTTestCase().test_data))
    #suite.addTest(TestRunner(CRTestCase().test_data))
    #suite.addTest(TestRunner(ShiftRotTestCase().test_data))
    #suite.addTest(TestRunner(LogicalTestCase().test_data))
    #suite.addTest(TestRunner(ALUTestCase().test_data))
    #suite.addTest(TestRunner(BranchTestCase().test_data))

    runner = unittest.TextTestRunner()
    runner.run(suite)
