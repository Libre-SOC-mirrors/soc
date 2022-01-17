"""TestRunner class, runs TestIssuer instructions

related bugs:

 * https://bugs.libre-soc.org/show_bug.cgi?id=363
 * https://bugs.libre-soc.org/show_bug.cgi?id=686#c51
"""
from nmigen import Module, Signal
from nmigen.hdl.xfrm import ResetInserter
from copy import copy
from pprint import pprint

# NOTE: to use cxxsim, export NMIGEN_SIM_MODE=cxxsim from the shell
# Also, check out the cxxsim nmigen branch, and latest yosys from git
from nmutil.sim_tmp_alternative import Simulator, Settle

from openpower.decoder.isa.caller import SVP64State
from openpower.decoder.isa.all import ISA
from openpower.endian import bigendian

from soc.simple.issuer import TestIssuerInternal
from soc.simple.inorder import TestIssuerInternalInOrder

from soc.simple.test.test_core import (setup_regs, check_regs, check_mem,
                                       wait_for_busy_clear,
                                       wait_for_busy_hi)
from soc.fu.compunits.test.test_compunit import (setup_tst_memory,
                                                 check_sim_memory)
from soc.debug.dmi import DBGCore, DBGCtrl, DBGStat
from nmutil.util import wrap
from openpower.test.state import TestState, StateRunner
from openpower.test.runner import TestRunnerBase


def insert_into_rom(startaddr, instructions, rom):
    print("insn before, init rom", len(instructions))
    pprint(rom)

    startaddr //= 4  # instructions are 32-bit

    # 64 bit
    mask = ((1 << 64)-1)
    for ins in instructions:
        if isinstance(ins, tuple):
            insn, code = ins
        else:
            insn, code = ins, ''
        insn = insn & 0xffffffff
        msbs = (startaddr >> 1) & mask
        lsb = 1 if (startaddr & 1) else 0
        print ("insn", hex(insn), hex(msbs), hex(lsb))

        val = rom.get(msbs<<3, 0)
        if insn != 0:
            print("before set", hex(4*startaddr),
                  hex(msbs), hex(val), hex(insn))
        val = (val | (insn << (lsb*32)))
        val = val & mask
        rom[msbs<<3] = val
        if insn != 0:
            print("after  set", hex(4*startaddr), hex(msbs), hex(val))
            print("instr: %06x 0x%x %s %08x" % (4*startaddr, insn, code, val))
        startaddr += 1
        startaddr = startaddr & mask

    print ("after insn insert")
    pprint(rom)


def setup_i_memory(imem, startaddr, instructions, rom):
    mem = imem
    print("insn before, init mem", mem.depth, mem.width, mem,
          len(instructions))

    if not rom:
        # initialise mem array to zero
        for i in range(mem.depth):
            yield mem._array[i].eq(0)
        yield Settle()

    startaddr //= 4  # instructions are 32-bit
    if mem.width == 32:
        assert rom is None, "cannot do 32-bit from wb_get ROM yet"
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
        lsb = 1 if (startaddr & 1) else 0

        if rom: # must put the value into the wb_get area
            val = rom[msbs<<1]
        else:
            val = yield mem._array[msbs]
        if insn != 0:
            print("before set", hex(4*startaddr),
                  hex(msbs), hex(val), hex(insn))
        val = (val | (insn << (lsb*32)))
        val = val & mask
        if rom: # must put the value into the wb_get area
            rom[msbs<<1] = val
        else:
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
    """HDLRunner:  Implements methods for the setup, preparation, and
    running of tests using nmigen HDL simulation.
    """

    def __init__(self, dut, m, pspec):
        super().__init__("hdl", HDLRunner)

        self.dut = dut
        self.pspec = pspec
        self.pc_i = Signal(32)
        self.svstate_i = Signal(64)

        #hard_reset = Signal(reset_less=True)
        if pspec.inorder:
            self.issuer = TestIssuerInternalInOrder(pspec)
        else:
            self.issuer = TestIssuerInternal(pspec)
        # use DMI RESET command instead, this does actually work though
        # issuer = ResetInserter({'coresync': hard_reset,
        #                        'sync': hard_reset})(issuer)
        m.submodules.issuer = self.issuer
        self.dmi = self.issuer.dbg.dmi

        comb = m.d.comb
        comb += self.issuer.pc_i.data.eq(self.pc_i)
        comb += self.issuer.svstate_i.data.eq(self.svstate_i)

    def prepare_for_test(self, test):
        self.test = test
        #print ("preparing for test name", test.name)

        # set up bigendian (TODO: don't do this, use MSR)
        yield self.issuer.core_bigendian_i.eq(bigendian)
        yield Settle()

        yield
        yield
        yield
        yield
        #print ("end of test preparation", test.name)

    def setup_during_test(self):
        # first run a manual hard-reset of the debug interface.
        # core is counting down on a 3-clock delay at this point
        yield self.issuer.dbg_rst_i.eq(1)
        yield
        yield self.issuer.dbg_rst_i.eq(0)

        # now run a DMI-interface reset.  because DMI is running
        # in dbgsync domain its reset is *NOT* connected to
        # core reset (hence the dbg_rst_i blip, above)
        yield from set_dmi(self.dmi, DBGCore.CTRL, 1 << DBGCtrl.STOP)
        yield
        #print("test setup")

    def run_test(self, instructions):
        """run_hdl_state - runs a TestIssuer nmigen HDL simulation
        """

        #print("starting test")

        if self.dut.rom is None:
            imem = self.issuer.imem._get_memory()
            #print("got memory", imem)
        else:
            print("skipping memory get due to rom")
            pprint(self.dut.rom)
        core = self.issuer.core
        dmi = self.issuer.dbg.dmi
        pdecode2 = self.issuer.pdecode2
        l0 = core.l0
        hdl_states = []

        # establish the TestIssuer context (mem, regs etc)

        pc = 0  # start address
        counter = 0  # test to pause/start

        # XXX for now, when ROM (run under wb_get) is detected,
        # skip setup of memories.  must be done a different way
        if self.dut.rom is None:
            yield from setup_i_memory(imem, pc, instructions, self.dut.rom)
            yield from setup_tst_memory(l0, self.test.mem)
        else:
            insert_into_rom(pc, instructions, self.dut.default_mem)
        print("about to setup regs")
        yield from setup_regs(pdecode2, core, self.test)
        #print("setup mem and regs done")

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

        # before starting the simulation, set the core stop address to be
        # just after the last instruction. if a load of an instruction is
        # requested at this address, the core is immediately put into "halt"
        # XXX: keep an eye out for in-order problems
        hard_stop_addr = self.test.stop_at_pc
        if hard_stop_addr is None:
            hard_stop_addr = len(instructions)*4
        yield from set_dmi(dmi, DBGCore.STOPADDR, hard_stop_addr)

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
                                   1 << DBGCtrl.START)
                yield self.issuer.pc_i.ok.eq(0)  # no change PC after this
                yield self.issuer.svstate_i.ok.eq(0)  # ditto
                yield
                yield

            counter = counter + 1

            # wait until executed
            while not ((yield self.issuer.insn_done) or
                       (yield self.issuer.dbg.terminated_o)):
                yield

            # okaaay long story: in overlap mode, PC is updated one cycle
            # late.
            if self.dut.allow_overlap:
                yield
            yield Settle()

            index = (yield self.issuer.cur_state.pc) // 4

            terminated = yield self.issuer.dbg.terminated_o
            print("terminated", terminated, index, len(instructions))

            if index < len(instructions):
                # Get HDL mem and state
                state = yield from TestState("hdl", core, self.dut,
                                             code)
                hdl_states.append(state)

            if index >= len(instructions):
                print("index over, send dmi stop")
                # stop at end
                yield from set_dmi(dmi, DBGCore.CTRL, 1 << DBGCtrl.STOP)
                yield
                yield
                # hmm really should use DMI status check here but hey it's quick
                while True:
                    stopped = yield self.issuer.dbg.core_stop_o
                    if stopped:
                        break
                    yield
                break

            terminated = yield self.issuer.dbg.terminated_o
            print("terminated(2)", terminated)
            if terminated:
                break

        if self.dut.allow_overlap: # or not self.dut.rom: ??
            # wait until all settled
            # XXX really this should be in DMI, which should in turn
            # use issuer.any_busy to not send back "stopped" signal
            while (yield self.issuer.any_busy):
                yield

        if self.dut.allow_overlap:
            # get last state, at end of run
            state = yield from TestState("hdl", core, self.dut,
                                         code)
            hdl_states.append(state)

        return hdl_states

    def end_test(self):
        yield from set_dmi(self.dmi, DBGCore.CTRL, 1 << DBGCtrl.STOP)
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

        # get MSR
        msr = yield from get_dmi(self.dmi, DBGCore.MSR)
        print("after test %s MSR value %x" % (self.test.name, msr))

        # test of dmi reg get
        for int_reg in range(32):
            yield from set_dmi(self.dmi, DBGCore.GSPR_IDX, int_reg)
            value = yield from get_dmi(self.dmi, DBGCore.GSPR_DATA)

            print("after test %s reg %2d value %x" %
                  (self.test.name, int_reg, value))

        # pull a reset
        yield from set_dmi(self.dmi, DBGCore.CTRL, 1 << DBGCtrl.RESET)
        yield


class TestRunner(TestRunnerBase):
    def __init__(self, tst_data, microwatt_mmu=False, rom=None,
                 svp64=True, inorder=False, run_hdl=True, run_sim=True,
                 allow_overlap=False):
        if run_hdl:
            run_hdl = HDLRunner
        super().__init__(tst_data, microwatt_mmu=microwatt_mmu,
                         rom=rom, inorder=inorder,
                         svp64=svp64, run_hdl=run_hdl, run_sim=run_sim,
                         allow_overlap=allow_overlap)
