"""TestRunner class, runs TestIssuer instructions

related bugs:

 * https://bugs.libre-soc.org/show_bug.cgi?id=363
 * https://bugs.libre-soc.org/show_bug.cgi?id=686#c51
"""
from nmigen import Module, Signal
from nmigen.hdl.xfrm import ResetInserter
from copy import copy

# NOTE: to use cxxsim, export NMIGEN_SIM_MODE=cxxsim from the shell
# Also, check out the cxxsim nmigen branch, and latest yosys from git
from nmutil.sim_tmp_alternative import Simulator, Settle

from openpower.decoder.isa.caller import SVP64State
from openpower.decoder.isa.all import ISA
from openpower.endian import bigendian

from soc.simple.issuer import TestIssuerInternal

from soc.config.test.test_loadstore import TestMemPspec
from soc.simple.test.test_core import (setup_regs, check_regs, check_mem,
                                       wait_for_busy_clear,
                                       wait_for_busy_hi)
from soc.fu.compunits.test.test_compunit import (setup_tst_memory,
                                                 check_sim_memory)
from soc.debug.dmi import DBGCore, DBGCtrl, DBGStat
from nmutil.util import wrap
from soc.experiment.test.test_mmu_dcache import wb_get
from openpower.test.state import TestState, StateRunner
from openpower.test.runner import TestRunnerBase


def setup_i_memory(imem, startaddr, instructions):
    mem = imem
    print("insn before, init mem", mem.depth, mem.width, mem,
          len(instructions))
    for i in range(mem.depth):
        yield mem._array[i].eq(0)
    yield Settle()
    startaddr //= 4  # instructions are 32-bit
    if mem.width == 32:
        mask = ((1 << 32)-1)
        for ins in instructions:
            if isinstance(ins, tuple):
                insn, code = ins
            else:
                insn, code = ins, ''
            insn = insn & 0xffffffff
            yield mem._array[startaddr].eq(insn)
            yield Settle()
            if insn != 0:
                print("instr: %06x 0x%x %s" % (4*startaddr, insn, code))
            startaddr += 1
            startaddr = startaddr & mask
        return

    # 64 bit
    mask = ((1 << 64)-1)
    for ins in instructions:
        if isinstance(ins, tuple):
            insn, code = ins
        else:
            insn, code = ins, ''
        insn = insn & 0xffffffff
        msbs = (startaddr >> 1) & mask
        val = yield mem._array[msbs]
        if insn != 0:
            print("before set", hex(4*startaddr),
                  hex(msbs), hex(val), hex(insn))
        lsb = 1 if (startaddr & 1) else 0
        val = (val | (insn << (lsb*32)))
        val = val & mask
        yield mem._array[msbs].eq(val)
        yield Settle()
        if insn != 0:
            print("after  set", hex(4*startaddr), hex(msbs), hex(val))
            print("instr: %06x 0x%x %s %08x" % (4*startaddr, insn, code, val))
        startaddr += 1
        startaddr = startaddr & mask


def set_dmi(dmi, addr, data):
    yield dmi.req_i.eq(1)
    yield dmi.addr_i.eq(addr)
    yield dmi.din.eq(data)
    yield dmi.we_i.eq(1)
    while True:
        ack = yield dmi.ack_o
        if ack:
            break
        yield
    yield
    yield dmi.req_i.eq(0)
    yield dmi.addr_i.eq(0)
    yield dmi.din.eq(0)
    yield dmi.we_i.eq(0)
    yield


def get_dmi(dmi, addr):
    yield dmi.req_i.eq(1)
    yield dmi.addr_i.eq(addr)
    yield dmi.din.eq(0)
    yield dmi.we_i.eq(0)
    while True:
        ack = yield dmi.ack_o
        if ack:
            break
        yield
    yield  # wait one
    data = yield dmi.dout  # get data after ack valid for 1 cycle
    yield dmi.req_i.eq(0)
    yield dmi.addr_i.eq(0)
    yield dmi.we_i.eq(0)
    yield
    return data


class HDLRunner(StateRunner):
    def __init__(self, dut, m, pspec):
        super().__init__("hdl", HDLRunner)

        self.dut = dut
        self.pc_i = Signal(32)
        self.svstate_i = Signal(64)

        #hard_reset = Signal(reset_less=True)
        self.issuer = TestIssuerInternal(pspec)
        # use DMI RESET command instead, this does actually work though
        #issuer = ResetInserter({'coresync': hard_reset,
        #                        'sync': hard_reset})(issuer)
        m.submodules.issuer = self.issuer
        self.dmi = self.issuer.dbg.dmi

        comb = m.d.comb
        comb += self.issuer.pc_i.data.eq(self.pc_i)
        comb += self.issuer.svstate_i.data.eq(self.svstate_i)

    def prepare_for_test(self, test):
        self.test = test

        # set up bigendian (TODO: don't do this, use MSR)
        yield self.issuer.core_bigendian_i.eq(bigendian)
        yield Settle()

        yield
        yield
        yield
        yield

    def setup_during_test(self):
        yield from set_dmi(self.dmi, DBGCore.CTRL, 1<<DBGCtrl.STOP)
        yield

    def run_test(self, instructions):
        """run_hdl_state - runs a TestIssuer nmigen HDL simulation
        """

        imem = self.issuer.imem._get_memory()
        core = self.issuer.core
        dmi = self.issuer.dbg.dmi
        pdecode2 = self.issuer.pdecode2
        l0 = core.l0
        hdl_states = []

        # establish the TestIssuer context (mem, regs etc)

        pc = 0  # start address
        counter = 0  # test to pause/start

        yield from setup_i_memory(imem, pc, instructions)
        yield from setup_tst_memory(l0, self.test.mem)
        yield from setup_regs(pdecode2, core, self.test)

        # set PC and SVSTATE
        yield self.pc_i.eq(pc)
        yield self.issuer.pc_i.ok.eq(1)

        # copy initial SVSTATE
        initial_svstate = copy(self.test.svstate)
        if isinstance(initial_svstate, int):
            initial_svstate = SVP64State(initial_svstate)
        yield self.svstate_i.eq(initial_svstate.value)
        yield self.issuer.svstate_i.ok.eq(1)
        yield

        print("instructions", instructions)

        # run the loop of the instructions on the current test
        index = (yield self.issuer.cur_state.pc) // 4
        while index < len(instructions):
            ins, code = instructions[index]

            print("hdl instr: 0x{:X}".format(ins & 0xffffffff))
            print(index, code)

            if counter == 0:
                # start the core
                yield
                yield from set_dmi(dmi, DBGCore.CTRL,
                                1<<DBGCtrl.START)
                yield self.issuer.pc_i.ok.eq(0) # no change PC after this
                yield self.issuer.svstate_i.ok.eq(0) # ditto
                yield
                yield

            counter = counter + 1

            # wait until executed
            while not (yield self.issuer.insn_done):
                yield

            yield Settle()

            index = (yield self.issuer.cur_state.pc) // 4

            terminated = yield self.issuer.dbg.terminated_o
            print("terminated", terminated)

            if index < len(instructions):
                # Get HDL mem and state
                state = yield from TestState("hdl", core, self.dut,
                                            code)
                hdl_states.append(state)

            if index >= len(instructions):
                print ("index over, send dmi stop")
                # stop at end
                yield from set_dmi(dmi, DBGCore.CTRL,
                                1<<DBGCtrl.STOP)
                yield
                yield

            terminated = yield self.issuer.dbg.terminated_o
            print("terminated(2)", terminated)
            if terminated:
                break

        return hdl_states

    def end_test(self):
        yield from set_dmi(self.dmi, DBGCore.CTRL, 1<<DBGCtrl.STOP)
        yield
        yield

        # TODO, here is where the static (expected) results
        # can be checked: register check (TODO, memory check)
        # see https://bugs.libre-soc.org/show_bug.cgi?id=686#c51
        # yield from check_regs(self, sim, core, test, code,
        #                       >>>expected_data<<<)

        # get CR
        cr = yield from get_dmi(self.dmi, DBGCore.CR)
        print("after test %s cr value %x" % (self.test.name, cr))

        # get XER
        xer = yield from get_dmi(self.dmi, DBGCore.XER)
        print("after test %s XER value %x" % (self.test.name, xer))

        # test of dmi reg get
        for int_reg in range(32):
            yield from set_dmi(self.dmi, DBGCore.GSPR_IDX, int_reg)
            value = yield from get_dmi(self.dmi, DBGCore.GSPR_DATA)

            print("after test %s reg %2d value %x" %
            (self.test.name, int_reg, value))

        # pull a reset
        yield from set_dmi(self.dmi, DBGCore.CTRL, 1<<DBGCtrl.RESET)
        yield


class TestRunner(TestRunnerBase):
    def __init__(self, tst_data, microwatt_mmu=False, rom=None,
                        svp64=True, run_hdl=True, run_sim=True):
        if run_hdl:
            run_hdl = HDLRunner
        super().__init__(tst_data, microwatt_mmu=microwatt_mmu,
                        rom=rom,
                        svp64=svp64, run_hdl=run_hdl, run_sim=run_sim)

