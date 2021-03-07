"""simple core issuer

not in any way intended for production use.  this runs a FSM that:

* reads the Program Counter from StateRegs
* reads an instruction from a fixed-size Test Memory
* issues it to the Simple Core
* waits for it to complete
* increments the PC
* does it all over again

the purpose of this module is to verify the functional correctness
of the Function Units in the absolute simplest and clearest possible
way, and to at provide something that can be further incrementally
improved.
"""

from nmigen import (Elaboratable, Module, Signal, ClockSignal, ResetSignal,
                    ClockDomain, DomainRenamer, Mux, Const)
from nmigen.cli import rtlil
from nmigen.cli import main
import sys

from soc.decoder.power_decoder import create_pdecode
from soc.decoder.power_decoder2 import PowerDecode2, SVP64PrefixDecoder
from soc.decoder.decode2execute1 import IssuerDecode2ToOperand
from soc.decoder.decode2execute1 import Data
from soc.experiment.testmem import TestMemory # test only for instructions
from soc.regfile.regfiles import StateRegs, FastRegs
from soc.simple.core import NonProductionCore
from soc.config.test.test_loadstore import TestMemPspec
from soc.config.ifetch import ConfigFetchUnit
from soc.decoder.power_enums import MicrOp
from soc.debug.dmi import CoreDebug, DMIInterface
from soc.debug.jtag import JTAG
from soc.config.pinouts import get_pinspecs
from soc.config.state import CoreState
from soc.interrupts.xics import XICS_ICP, XICS_ICS
from soc.bus.simple_gpio import SimpleGPIO
from soc.bus.SPBlock512W64B8W import SPBlock512W64B8W
from soc.clock.select import ClockSelect
from soc.clock.dummypll import DummyPLL
from soc.sv.svstate import SVSTATERec


from nmutil.util import rising_edge

def get_insn(f_instr_o, pc):
    if f_instr_o.width == 32:
        return f_instr_o
    else:
        # 64-bit: bit 2 of pc decides which word to select
        return f_instr_o.word_select(pc[2], 32)


class TestIssuerInternal(Elaboratable):
    """TestIssuer - reads instructions from TestMemory and issues them

    efficiency and speed is not the main goal here: functional correctness is.
    """
    def __init__(self, pspec):

        # JTAG interface.  add this right at the start because if it's
        # added it *modifies* the pspec, by adding enable/disable signals
        # for parts of the rest of the core
        self.jtag_en = hasattr(pspec, "debug") and pspec.debug == 'jtag'
        if self.jtag_en:
            subset = {'uart', 'mtwi', 'eint', 'gpio', 'mspi0', 'mspi1',
                      'pwm', 'sd0', 'sdr'}
            self.jtag = JTAG(get_pinspecs(subset=subset))
            # add signals to pspec to enable/disable icache and dcache
            # (or data and intstruction wishbone if icache/dcache not included)
            # https://bugs.libre-soc.org/show_bug.cgi?id=520
            # TODO: do we actually care if these are not domain-synchronised?
            # honestly probably not.
            pspec.wb_icache_en = self.jtag.wb_icache_en
            pspec.wb_dcache_en = self.jtag.wb_dcache_en
            self.wb_sram_en = self.jtag.wb_sram_en
        else:
            self.wb_sram_en = Const(1)

        # add 4k sram blocks?
        self.sram4x4k = (hasattr(pspec, "sram4x4kblock") and
                         pspec.sram4x4kblock == True)
        if self.sram4x4k:
            self.sram4k = []
            for i in range(4):
                self.sram4k.append(SPBlock512W64B8W(name="sram4k_%d" % i,
                                                    features={'err'}))

        # add interrupt controller?
        self.xics = hasattr(pspec, "xics") and pspec.xics == True
        if self.xics:
            self.xics_icp = XICS_ICP()
            self.xics_ics = XICS_ICS()
            self.int_level_i = self.xics_ics.int_level_i

        # add GPIO peripheral?
        self.gpio = hasattr(pspec, "gpio") and pspec.gpio == True
        if self.gpio:
            self.simple_gpio = SimpleGPIO()
            self.gpio_o = self.simple_gpio.gpio_o

        # main instruction core25
        self.core = core = NonProductionCore(pspec)

        # instruction decoder.  goes into Trap Record
        pdecode = create_pdecode()
        self.cur_state = CoreState("cur") # current state (MSR/PC/EINT/SVSTATE)
        self.pdecode2 = PowerDecode2(pdecode, state=self.cur_state,
                                     opkls=IssuerDecode2ToOperand)
        self.svp64 = SVP64PrefixDecoder() # for decoding SVP64 prefix

        # Test Instruction memory
        self.imem = ConfigFetchUnit(pspec).fu
        # one-row cache of instruction read
        self.iline = Signal(64) # one instruction line
        self.iprev_adr = Signal(64) # previous address: if different, do read

        # DMI interface
        self.dbg = CoreDebug()

        # instruction go/monitor
        self.pc_o = Signal(64, reset_less=True)
        self.pc_i = Data(64, "pc_i") # set "ok" to indicate "please change me"
        self.svstate_i = Data(32, "svstate_i") # ditto
        self.core_bigendian_i = Signal()
        self.busy_o = Signal(reset_less=True)
        self.memerr_o = Signal(reset_less=True)

        # STATE regfile read /write ports for PC, MSR, SVSTATE
        staterf = self.core.regs.rf['state']
        self.state_r_pc = staterf.r_ports['cia'] # PC rd
        self.state_w_pc = staterf.w_ports['d_wr1'] # PC wr
        self.state_r_msr = staterf.r_ports['msr'] # MSR rd
        self.state_r_sv = staterf.r_ports['sv'] # SVSTATE rd
        self.state_w_sv = staterf.w_ports['sv'] # SVSTATE wr

        # DMI interface access
        intrf = self.core.regs.rf['int']
        crrf = self.core.regs.rf['cr']
        xerrf = self.core.regs.rf['xer']
        self.int_r = intrf.r_ports['dmi'] # INT read
        self.cr_r = crrf.r_ports['full_cr_dbg'] # CR read
        self.xer_r = xerrf.r_ports['full_xer'] # XER read

        # hack method of keeping an eye on whether branch/trap set the PC
        self.state_nia = self.core.regs.rf['state'].w_ports['nia']
        self.state_nia.wen.name = 'state_nia_wen'

    def fetch_fsm(self, m, core, pc, svstate, nia,
                        fetch_pc_ready_o, fetch_pc_valid_i,
                        fetch_insn_valid_o, fetch_insn_ready_i):
        """fetch FSM
        this FSM performs fetch of raw instruction data, partial-decodes
        it 32-bit at a time to detect SVP64 prefixes, and will optionally
        read a 2nd 32-bit quantity if that occurs.
        """
        comb = m.d.comb
        sync = m.d.sync
        pdecode2 = self.pdecode2
        svp64 = self.svp64
        cur_state = self.cur_state
        dec_opcode_i = pdecode2.dec.raw_opcode_in # raw opcode

        msr_read = Signal(reset=1)

        with m.FSM(name='fetch_fsm'):

            # waiting (zzz)
            with m.State("IDLE"):
                comb += fetch_pc_ready_o.eq(1)
                with m.If(fetch_pc_valid_i):
                    # instruction allowed to go: start by reading the PC
                    # capture the PC and also drop it into Insn Memory
                    # we have joined a pair of combinatorial memory
                    # lookups together.  this is Generally Bad.
                    comb += self.imem.a_pc_i.eq(pc)
                    comb += self.imem.a_valid_i.eq(1)
                    comb += self.imem.f_valid_i.eq(1)
                    sync += cur_state.pc.eq(pc)
                    sync += cur_state.svstate.eq(svstate) # and svstate

                    # initiate read of MSR. arrives one clock later
                    comb += self.state_r_msr.ren.eq(1 << StateRegs.MSR)
                    sync += msr_read.eq(0)

                    m.next = "INSN_READ"  # move to "wait for bus" phase

            # dummy pause to find out why simulation is not keeping up
            with m.State("INSN_READ"):
                # one cycle later, msr/sv read arrives.  valid only once.
                with m.If(~msr_read):
                    sync += msr_read.eq(1) # yeah don't read it again
                    sync += cur_state.msr.eq(self.state_r_msr.data_o)
                with m.If(self.imem.f_busy_o): # zzz...
                    # busy: stay in wait-read
                    comb += self.imem.a_valid_i.eq(1)
                    comb += self.imem.f_valid_i.eq(1)
                with m.Else():
                    # not busy: instruction fetched
                    insn = get_insn(self.imem.f_instr_o, cur_state.pc)
                    # decode the SVP64 prefix, if any
                    comb += svp64.raw_opcode_in.eq(insn)
                    comb += svp64.bigendian.eq(self.core_bigendian_i)
                    # pass the decoded prefix (if any) to PowerDecoder2
                    sync += pdecode2.sv_rm.eq(svp64.svp64_rm)
                    # calculate the address of the following instruction
                    insn_size = Mux(svp64.is_svp64_mode, 8, 4)
                    sync += nia.eq(cur_state.pc + insn_size)
                    with m.If(~svp64.is_svp64_mode):
                        # with no prefix, store the instruction
                        # and hand it directly to the next FSM
                        sync += dec_opcode_i.eq(insn)
                        m.next = "INSN_READY"
                    with m.Else():
                        # fetch the rest of the instruction from memory
                        comb += self.imem.a_pc_i.eq(cur_state.pc + 4)
                        comb += self.imem.a_valid_i.eq(1)
                        comb += self.imem.f_valid_i.eq(1)
                        m.next = "INSN_READ2"

            with m.State("INSN_READ2"):
                with m.If(self.imem.f_busy_o):  # zzz...
                    # busy: stay in wait-read
                    comb += self.imem.a_valid_i.eq(1)
                    comb += self.imem.f_valid_i.eq(1)
                with m.Else():
                    # not busy: instruction fetched
                    insn = get_insn(self.imem.f_instr_o, cur_state.pc+4)
                    sync += dec_opcode_i.eq(insn)
                    m.next = "INSN_READY"

            with m.State("INSN_READY"):
                # hand over the instruction, to be decoded
                comb += fetch_insn_valid_o.eq(1)
                with m.If(fetch_insn_ready_i):
                    m.next = "IDLE"

    def issue_fsm(self, m, core, pc_changed, sv_changed, nia,
                  dbg, core_rst,
                  fetch_pc_ready_o, fetch_pc_valid_i,
                  fetch_insn_valid_o, fetch_insn_ready_i,
                  exec_insn_valid_i, exec_insn_ready_o,
                  exec_pc_valid_o, exec_pc_ready_i):
        """issue FSM

        decode / issue FSM.  this interacts with the "fetch" FSM
        through fetch_insn_ready/valid (incoming) and fetch_pc_ready/valid
        (outgoing). also interacts with the "execute" FSM
        through exec_insn_ready/valid (outgoing) and exec_pc_ready/valid
        (incoming).
        SVP64 RM prefixes have already been set up by the
        "fetch" phase, so execute is fairly straightforward.
        """

        comb = m.d.comb
        sync = m.d.sync
        pdecode2 = self.pdecode2
        cur_state = self.cur_state

        # temporaries
        dec_opcode_i = pdecode2.dec.raw_opcode_in # raw opcode

        # for updating svstate (things like srcstep etc.)
        update_svstate = Signal() # set this (below) if updating
        new_svstate = SVSTATERec("new_svstate")
        comb += new_svstate.eq(cur_state.svstate)

        with m.FSM(name="issue_fsm"):

            # Wait on "core stop" release, at reset
            with m.State("WAIT_RESET"):
                with m.If(~dbg.core_stop_o & ~core_rst):
                    m.next = "INSN_FETCH"
                with m.Else():
                    comb += core.core_stopped_i.eq(1)
                    comb += dbg.core_stopped_i.eq(1)
                    # while stopped, allow updating the PC and SVSTATE
                    with m.If(self.pc_i.ok):
                        comb += self.state_w_pc.wen.eq(1 << StateRegs.PC)
                        comb += self.state_w_pc.data_i.eq(self.pc_i.data)
                        sync += pc_changed.eq(1)
                    with m.If(self.svstate_i.ok):
                        comb += new_svstate.eq(self.svstate_i.data)
                        comb += update_svstate.eq(1)
                        sync += sv_changed.eq(1)

            # go fetch the instruction at the current PC
            # at this point, there is no instruction running, that
            # could inadvertently update the PC.
            with m.State("INSN_FETCH"):
                # TODO: update PC here, before fetch
                comb += fetch_pc_valid_i.eq(1)
                with m.If(fetch_pc_ready_o):
                    m.next = "INSN_WAIT"

            # decode the instruction when it arrives
            with m.State("INSN_WAIT"):
                comb += fetch_insn_ready_i.eq(1)
                with m.If(fetch_insn_valid_o):
                    # decode the instruction
                    sync += core.e.eq(pdecode2.e)
                    sync += core.state.eq(cur_state)
                    sync += core.raw_insn_i.eq(dec_opcode_i)
                    sync += core.bigendian_i.eq(self.core_bigendian_i)
                    # loop into INSN_FETCH if it's a vector instruction
                    # and VL == 0.  this because VL==0 is a for-loop
                    # from 0 to 0 i.e. always, always a NOP.
                    cur_vl = cur_state.svstate.vl
                    with m.If(~pdecode2.no_out_vec & (cur_vl == 0)):
                        # update the PC before fetching the next instruction
                        # since we are in a VL==0 loop, no instruction was
                        # executed that we could be overwriting
                        comb += self.state_w_pc.wen.eq(1 << StateRegs.PC)
                        comb += self.state_w_pc.data_i.eq(nia)
                        m.next = "INSN_FETCH"
                    with m.Else():
                        m.next = "INSN_EXECUTE"  # move to "execute"

            with m.State("INSN_EXECUTE"):
                comb += exec_insn_valid_i.eq(1)
                with m.If(exec_insn_ready_o):
                    m.next = "EXECUTE_WAIT"

            with m.State("EXECUTE_WAIT"):
                # wait on "core stop" release, at instruction end
                with m.If(~dbg.core_stop_o & ~core_rst):
                    comb += exec_pc_ready_i.eq(1)
                    with m.If(exec_pc_valid_o):
                        # precalculate srcstep+1
                        next_srcstep = Signal.like(cur_state.svstate.srcstep)
                        comb += next_srcstep.eq(cur_state.svstate.srcstep+1)
                        # was this the last loop iteration?
                        is_last = Signal()
                        cur_vl = cur_state.svstate.vl
                        comb += is_last.eq(next_srcstep == cur_vl)

                        # if either PC or SVSTATE were changed by the previous
                        # instruction, go directly back to Fetch, without
                        # updating either PC or SVSTATE
                        with m.If(pc_changed | sv_changed):
                            m.next = "INSN_FETCH"

                        # also return to Fetch, when no output was a vector
                        # (regardless of SRCSTEP and VL), or when the last
                        # instruction was really the last one of the VL loop
                        with m.Elif(pdecode2.no_out_vec | is_last):
                            # before going back to fetch, update the PC state
                            # register with the NIA.
                            # ok here we are not reading the branch unit.
                            # TODO: this just blithely overwrites whatever
                            #       pipeline updated the PC
                            comb += self.state_w_pc.wen.eq(1 << StateRegs.PC)
                            comb += self.state_w_pc.data_i.eq(nia)
                            # reset SRCSTEP before returning to Fetch
                            with m.If(~pdecode2.no_out_vec):
                                comb += new_svstate.srcstep.eq(0)
                                comb += update_svstate.eq(1)
                            m.next = "INSN_FETCH"

                        # returning to Execute? then, first update SRCSTEP
                        with m.Else():
                            comb += new_svstate.srcstep.eq(next_srcstep)
                            comb += update_svstate.eq(1)
                            m.next = "DECODE_SV"

                with m.Else():
                    comb += core.core_stopped_i.eq(1)
                    comb += dbg.core_stopped_i.eq(1)
                    # while stopped, allow updating the PC and SVSTATE
                    with m.If(self.pc_i.ok):
                        comb += self.state_w_pc.wen.eq(1 << StateRegs.PC)
                        comb += self.state_w_pc.data_i.eq(self.pc_i.data)
                        sync += pc_changed.eq(1)
                    with m.If(self.svstate_i.ok):
                        comb += new_svstate.eq(self.svstate_i.data)
                        comb += update_svstate.eq(1)
                        sync += sv_changed.eq(1)

            # need to decode the instruction again, after updating SRCSTEP
            # in the previous state.
            # mostly a copy of INSN_WAIT, but without the actual wait
            with m.State("DECODE_SV"):
                # decode the instruction
                sync += core.e.eq(pdecode2.e)
                sync += core.state.eq(cur_state)
                sync += core.bigendian_i.eq(self.core_bigendian_i)
                m.next = "INSN_EXECUTE"  # move to "execute"

        # check if svstate needs updating: if so, write it to State Regfile
        with m.If(update_svstate):
            comb += self.state_w_sv.wen.eq(1<<StateRegs.SVSTATE)
            comb += self.state_w_sv.data_i.eq(new_svstate)
            sync += cur_state.svstate.eq(new_svstate) # for next clock

    def execute_fsm(self, m, core, insn_done, pc_changed, sv_changed,
                    exec_insn_valid_i, exec_insn_ready_o,
                    exec_pc_valid_o, exec_pc_ready_i):
        """execute FSM

        execute FSM. this interacts with the "issue" FSM
        through exec_insn_ready/valid (incoming) and exec_pc_ready/valid
        (outgoing). SVP64 RM prefixes have already been set up by the
        "issue" phase, so execute is fairly straightforward.
        """

        comb = m.d.comb
        sync = m.d.sync
        pdecode2 = self.pdecode2
        svp64 = self.svp64

        # temporaries
        core_busy_o = core.busy_o                 # core is busy
        core_ivalid_i = core.ivalid_i             # instruction is valid
        core_issue_i = core.issue_i               # instruction is issued
        insn_type = core.e.do.insn_type           # instruction MicroOp type

        with m.FSM(name="exec_fsm"):

            # waiting for instruction bus (stays there until not busy)
            with m.State("INSN_START"):
                comb += exec_insn_ready_o.eq(1)
                with m.If(exec_insn_valid_i):
                    comb += core_ivalid_i.eq(1)  # instruction is valid
                    comb += core_issue_i.eq(1)  # and issued
                    sync += sv_changed.eq(0)
                    sync += pc_changed.eq(0)
                    m.next = "INSN_ACTIVE"  # move to "wait completion"

            # instruction started: must wait till it finishes
            with m.State("INSN_ACTIVE"):
                with m.If(insn_type != MicrOp.OP_NOP):
                    comb += core_ivalid_i.eq(1) # instruction is valid
                # note changes to PC and SVSTATE
                with m.If(self.state_nia.wen & (1<<StateRegs.SVSTATE)):
                    sync += sv_changed.eq(1)
                with m.If(self.state_nia.wen & (1<<StateRegs.PC)):
                    sync += pc_changed.eq(1)
                with m.If(~core_busy_o): # instruction done!
                    comb += insn_done.eq(1)
                    comb += exec_pc_valid_o.eq(1)
                    with m.If(exec_pc_ready_i):
                        m.next = "INSN_START"  # back to fetch

    def elaborate(self, platform):
        m = Module()
        comb, sync = m.d.comb, m.d.sync

        m.submodules.core = core = DomainRenamer("coresync")(self.core)
        m.submodules.imem = imem = self.imem
        m.submodules.dbg = dbg = self.dbg
        if self.jtag_en:
            m.submodules.jtag = jtag = self.jtag
            # TODO: UART2GDB mux, here, from external pin
            # see https://bugs.libre-soc.org/show_bug.cgi?id=499
            sync += dbg.dmi.connect_to(jtag.dmi)

        cur_state = self.cur_state

        # 4x 4k SRAM blocks.  these simply "exist", they get routed in litex
        if self.sram4x4k:
            for i, sram in enumerate(self.sram4k):
                m.submodules["sram4k_%d" % i] = sram
                comb += sram.enable.eq(self.wb_sram_en)

        # XICS interrupt handler
        if self.xics:
            m.submodules.xics_icp = icp = self.xics_icp
            m.submodules.xics_ics = ics = self.xics_ics
            comb += icp.ics_i.eq(ics.icp_o)           # connect ICS to ICP
            sync += cur_state.eint.eq(icp.core_irq_o) # connect ICP to core

        # GPIO test peripheral
        if self.gpio:
            m.submodules.simple_gpio = simple_gpio = self.simple_gpio

        # connect one GPIO output to ICS bit 15 (like in microwatt soc.vhdl)
        # XXX causes litex ECP5 test to get wrong idea about input and output
        # (but works with verilator sim *sigh*)
        #if self.gpio and self.xics:
        #   comb += self.int_level_i[15].eq(simple_gpio.gpio_o[0])

        # instruction decoder
        pdecode = create_pdecode()
        m.submodules.dec2 = pdecode2 = self.pdecode2
        m.submodules.svp64 = svp64 = self.svp64

        # convenience
        dmi, d_reg, d_cr, d_xer, = dbg.dmi, dbg.d_gpr, dbg.d_cr, dbg.d_xer
        intrf = self.core.regs.rf['int']

        # clock delay power-on reset
        cd_por  = ClockDomain(reset_less=True)
        cd_sync = ClockDomain()
        core_sync = ClockDomain("coresync")
        m.domains += cd_por, cd_sync, core_sync

        ti_rst = Signal(reset_less=True)
        delay = Signal(range(4), reset=3)
        with m.If(delay != 0):
            m.d.por += delay.eq(delay - 1)
        comb += cd_por.clk.eq(ClockSignal())

        # power-on reset delay
        core_rst = ResetSignal("coresync")
        comb += ti_rst.eq(delay != 0 | dbg.core_rst_o | ResetSignal())
        comb += core_rst.eq(ti_rst)

        # busy/halted signals from core
        comb += self.busy_o.eq(core.busy_o)
        comb += pdecode2.dec.bigendian.eq(self.core_bigendian_i)

        # temporary hack: says "go" immediately for both address gen and ST
        l0 = core.l0
        ldst = core.fus.fus['ldst0']
        st_go_edge = rising_edge(m, ldst.st.rel_o)
        m.d.comb += ldst.ad.go_i.eq(ldst.ad.rel_o) # link addr-go direct to rel
        m.d.comb += ldst.st.go_i.eq(st_go_edge) # link store-go to rising rel

        # PC and instruction from I-Memory
        comb += self.pc_o.eq(cur_state.pc)
        pc_changed = Signal() # note write to PC
        sv_changed = Signal() # note write to SVSTATE
        insn_done = Signal()  # fires just once

        # read the PC
        pc = Signal(64, reset_less=True)
        pc_ok_delay = Signal()
        sync += pc_ok_delay.eq(~self.pc_i.ok)
        with m.If(self.pc_i.ok):
            # incoming override (start from pc_i)
            comb += pc.eq(self.pc_i.data)
        with m.Else():
            # otherwise read StateRegs regfile for PC...
            comb += self.state_r_pc.ren.eq(1<<StateRegs.PC)
        # ... but on a 1-clock delay
        with m.If(pc_ok_delay):
            comb += pc.eq(self.state_r_pc.data_o)

        # read svstate
        svstate = Signal(64, reset_less=True)
        svstate_ok_delay = Signal()
        sync += svstate_ok_delay.eq(~self.svstate_i.ok)
        with m.If(self.svstate_i.ok):
            # incoming override (start from svstate__i)
            comb += svstate.eq(self.svstate_i.data)
        with m.Else():
            # otherwise read StateRegs regfile for SVSTATE...
            comb += self.state_r_sv.ren.eq(1 << StateRegs.SVSTATE)
        # ... but on a 1-clock delay
        with m.If(svstate_ok_delay):
            comb += svstate.eq(self.state_r_sv.data_o)

        # don't write pc every cycle
        comb += self.state_w_pc.wen.eq(0)
        comb += self.state_w_pc.data_i.eq(0)

        # don't read msr every cycle
        comb += self.state_r_msr.ren.eq(0)

        # address of the next instruction, in the absence of a branch
        # depends on the instruction size
        nia = Signal(64, reset_less=True)

        # connect up debug signals
        # TODO comb += core.icache_rst_i.eq(dbg.icache_rst_o)
        comb += dbg.terminate_i.eq(core.core_terminate_o)
        comb += dbg.state.pc.eq(pc)
        comb += dbg.state.svstate.eq(svstate)
        comb += dbg.state.msr.eq(cur_state.msr)

        # there are *TWO* FSMs, one fetch (32/64-bit) one decode/execute.
        # these are the handshake signals between fetch and decode/execute

        # fetch FSM can run as soon as the PC is valid
        fetch_pc_valid_i = Signal() # Execute tells Fetch "start next read"
        fetch_pc_ready_o = Signal() # Fetch Tells SVSTATE "proceed"

        # fetch FSM hands over the instruction to be decoded / issued
        fetch_insn_valid_o = Signal()
        fetch_insn_ready_i = Signal()

        # issue FSM delivers the instruction to the be executed
        exec_insn_valid_i = Signal()
        exec_insn_ready_o = Signal()

        # execute FSM, hands over the PC/SVSTATE back to the issue FSM
        exec_pc_valid_o = Signal()
        exec_pc_ready_i = Signal()

        # actually use a nmigen FSM for the first time (w00t)
        # this FSM is perhaps unusual in that it detects conditions
        # then "holds" information, combinatorially, for the core
        # (as opposed to using sync - which would be on a clock's delay)
        # this includes the actual opcode, valid flags and so on.

        self.fetch_fsm(m, core, pc, svstate, nia,
                       fetch_pc_ready_o, fetch_pc_valid_i,
                       fetch_insn_valid_o, fetch_insn_ready_i)

        # TODO: an SVSTATE-based for-loop FSM that goes in between
        # fetch pc/insn ready/valid and advances SVSTATE.srcstep
        # until it reaches VL-1 or PowerDecoder2.no_out_vec is True.
        self.issue_fsm(m, core, pc_changed, sv_changed, nia,
                       dbg, core_rst,
                       fetch_pc_ready_o, fetch_pc_valid_i,
                       fetch_insn_valid_o, fetch_insn_ready_i,
                       exec_insn_valid_i, exec_insn_ready_o,
                       exec_pc_ready_i, exec_pc_valid_o)

        self.execute_fsm(m, core, insn_done, pc_changed, sv_changed,
                         exec_insn_valid_i, exec_insn_ready_o,
                         exec_pc_ready_i, exec_pc_valid_o)

        # this bit doesn't have to be in the FSM: connect up to read
        # regfiles on demand from DMI
        with m.If(d_reg.req): # request for regfile access being made
            # TODO: error-check this
            # XXX should this be combinatorial?  sync better?
            if intrf.unary:
                comb += self.int_r.ren.eq(1<<d_reg.addr)
            else:
                comb += self.int_r.addr.eq(d_reg.addr)
                comb += self.int_r.ren.eq(1)
        d_reg_delay  = Signal()
        sync += d_reg_delay.eq(d_reg.req)
        with m.If(d_reg_delay):
            # data arrives one clock later
            comb += d_reg.data.eq(self.int_r.data_o)
            comb += d_reg.ack.eq(1)

        # sigh same thing for CR debug
        with m.If(d_cr.req): # request for regfile access being made
            comb += self.cr_r.ren.eq(0b11111111) # enable all
        d_cr_delay  = Signal()
        sync += d_cr_delay.eq(d_cr.req)
        with m.If(d_cr_delay):
            # data arrives one clock later
            comb += d_cr.data.eq(self.cr_r.data_o)
            comb += d_cr.ack.eq(1)

        # aaand XER...
        with m.If(d_xer.req): # request for regfile access being made
            comb += self.xer_r.ren.eq(0b111111) # enable all
        d_xer_delay  = Signal()
        sync += d_xer_delay.eq(d_xer.req)
        with m.If(d_xer_delay):
            # data arrives one clock later
            comb += d_xer.data.eq(self.xer_r.data_o)
            comb += d_xer.ack.eq(1)

        # DEC and TB inc/dec FSM.  copy of DEC is put into CoreState,
        # (which uses that in PowerDecoder2 to raise 0x900 exception)
        self.tb_dec_fsm(m, cur_state.dec)

        return m

    def tb_dec_fsm(self, m, spr_dec):
        """tb_dec_fsm

        this is a FSM for updating either dec or tb.  it runs alternately
        DEC, TB, DEC, TB.  note that SPR pipeline could have written a new
        value to DEC, however the regfile has "passthrough" on it so this
        *should* be ok.

        see v3.0B p1097-1099 for Timeer Resource and p1065 and p1076
        """

        comb, sync = m.d.comb, m.d.sync
        fast_rf = self.core.regs.rf['fast']
        fast_r_dectb = fast_rf.r_ports['issue'] # DEC/TB
        fast_w_dectb = fast_rf.w_ports['issue'] # DEC/TB

        with m.FSM() as fsm:

            # initiates read of current DEC
            with m.State("DEC_READ"):
                comb += fast_r_dectb.addr.eq(FastRegs.DEC)
                comb += fast_r_dectb.ren.eq(1)
                m.next = "DEC_WRITE"

            # waits for DEC read to arrive (1 cycle), updates with new value
            with m.State("DEC_WRITE"):
                new_dec = Signal(64)
                # TODO: MSR.LPCR 32-bit decrement mode
                comb += new_dec.eq(fast_r_dectb.data_o - 1)
                comb += fast_w_dectb.addr.eq(FastRegs.DEC)
                comb += fast_w_dectb.wen.eq(1)
                comb += fast_w_dectb.data_i.eq(new_dec)
                sync += spr_dec.eq(new_dec) # copy into cur_state for decoder
                m.next = "TB_READ"

            # initiates read of current TB
            with m.State("TB_READ"):
                comb += fast_r_dectb.addr.eq(FastRegs.TB)
                comb += fast_r_dectb.ren.eq(1)
                m.next = "TB_WRITE"

            # waits for read TB to arrive, initiates write of current TB
            with m.State("TB_WRITE"):
                new_tb = Signal(64)
                comb += new_tb.eq(fast_r_dectb.data_o + 1)
                comb += fast_w_dectb.addr.eq(FastRegs.TB)
                comb += fast_w_dectb.wen.eq(1)
                comb += fast_w_dectb.data_i.eq(new_tb)
                m.next = "DEC_READ"

        return m

    def __iter__(self):
        yield from self.pc_i.ports()
        yield self.pc_o
        yield self.memerr_o
        yield from self.core.ports()
        yield from self.imem.ports()
        yield self.core_bigendian_i
        yield self.busy_o

    def ports(self):
        return list(self)

    def external_ports(self):
        ports = self.pc_i.ports()
        ports += [self.pc_o, self.memerr_o, self.core_bigendian_i, self.busy_o,
                ]

        if self.jtag_en:
            ports += list(self.jtag.external_ports())
        else:
            # don't add DMI if JTAG is enabled
            ports += list(self.dbg.dmi.ports())

        ports += list(self.imem.ibus.fields.values())
        ports += list(self.core.l0.cmpi.lsmem.lsi.slavebus.fields.values())

        if self.sram4x4k:
            for sram in self.sram4k:
                ports += list(sram.bus.fields.values())

        if self.xics:
            ports += list(self.xics_icp.bus.fields.values())
            ports += list(self.xics_ics.bus.fields.values())
            ports.append(self.int_level_i)

        if self.gpio:
            ports += list(self.simple_gpio.bus.fields.values())
            ports.append(self.gpio_o)

        return ports

    def ports(self):
        return list(self)


class TestIssuer(Elaboratable):
    def __init__(self, pspec):
        self.ti = TestIssuerInternal(pspec)

        self.pll = DummyPLL()

        # PLL direct clock or not
        self.pll_en = hasattr(pspec, "use_pll") and pspec.use_pll
        if self.pll_en:
            self.pll_18_o = Signal(reset_less=True)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        # TestIssuer runs at direct clock
        m.submodules.ti = ti = self.ti
        cd_int = ClockDomain("coresync")

        if self.pll_en:
            # ClockSelect runs at PLL output internal clock rate
            m.submodules.pll = pll = self.pll

            # add clock domains from PLL
            cd_pll = ClockDomain("pllclk")
            m.domains += cd_pll

            # PLL clock established.  has the side-effect of running clklsel
            # at the PLL's speed (see DomainRenamer("pllclk") above)
            pllclk = ClockSignal("pllclk")
            comb += pllclk.eq(pll.clk_pll_o)

            # wire up external 24mhz to PLL
            comb += pll.clk_24_i.eq(ClockSignal())

            # output 18 mhz PLL test signal
            comb += self.pll_18_o.eq(pll.pll_18_o)

            # now wire up ResetSignals.  don't mind them being in this domain
            pll_rst = ResetSignal("pllclk")
            comb += pll_rst.eq(ResetSignal())

        # internal clock is set to selector clock-out.  has the side-effect of
        # running TestIssuer at this speed (see DomainRenamer("intclk") above)
        intclk = ClockSignal("coresync")
        if self.pll_en:
            comb += intclk.eq(pll.clk_pll_o)
        else:
            comb += intclk.eq(ClockSignal())

        return m

    def ports(self):
        return list(self.ti.ports()) + list(self.pll.ports()) + \
               [ClockSignal(), ResetSignal()]

    def external_ports(self):
        ports = self.ti.external_ports()
        ports.append(ClockSignal())
        ports.append(ResetSignal())
        if self.pll_en:
            ports.append(self.pll.clk_sel_i)
            ports.append(self.pll_18_o)
            ports.append(self.pll.pll_lck_o)
        return ports


if __name__ == '__main__':
    units = {'alu': 1, 'cr': 1, 'branch': 1, 'trap': 1, 'logical': 1,
             'spr': 1,
             'div': 1,
             'mul': 1,
             'shiftrot': 1
            }
    pspec = TestMemPspec(ldst_ifacetype='bare_wb',
                         imem_ifacetype='bare_wb',
                         addr_wid=48,
                         mask_wid=8,
                         reg_wid=64,
                         units=units)
    dut = TestIssuer(pspec)
    vl = main(dut, ports=dut.ports(), name="test_issuer")

    if len(sys.argv) == 1:
        vl = rtlil.convert(dut, ports=dut.external_ports(), name="test_issuer")
        with open("test_issuer.il", "w") as f:
            f.write(vl)
