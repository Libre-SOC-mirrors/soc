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

from nmigen import (Elaboratable, Module, Signal,
                    Mux, Const, Repl, Cat)
from nmigen.cli import rtlil
from nmigen.cli import main
import sys

from nmutil.singlepipe import ControlBase
from soc.simple.core_data import FetchOutput, FetchInput

from openpower.consts import MSR
from openpower.decoder.power_enums import MicrOp
from openpower.state import CoreState
from soc.regfile.regfiles import StateRegs
from soc.config.test.test_loadstore import TestMemPspec
from soc.experiment.icache import ICache

from nmutil.util import rising_edge

from soc.simple.issuer import TestIssuerBase

def get_insn(f_instr_o, pc):
    if f_instr_o.width == 32:
        return f_instr_o
    else:
        # 64-bit: bit 2 of pc decides which word to select
        return f_instr_o.word_select(pc[2], 32)


# Fetch Finite State Machine.
# WARNING: there are currently DriverConflicts but it's actually working.
# TODO, here: everything that is global in nature, information from the
# main TestIssuerInternal, needs to move to either ispec() or ospec().
# not only that: TestIssuerInternal.imem can entirely move into here
# because imem is only ever accessed inside the FetchFSM.
class FetchFSM(ControlBase):
    def __init__(self, allow_overlap, imem, core_rst,
                 pdecode2, cur_state,
                 dbg, core, svstate, nia):
        self.allow_overlap = allow_overlap
        self.imem = imem
        self.core_rst = core_rst
        self.pdecode2 = pdecode2
        self.cur_state = cur_state
        self.dbg = dbg
        self.core = core
        self.svstate = svstate
        self.nia = nia

        # set up pipeline ControlBase and allocate i/o specs
        # (unusual: normally done by the Pipeline API)
        super().__init__(stage=self)
        self.p.i_data, self.n.o_data = self.new_specs(None)
        self.i, self.o = self.p.i_data, self.n.o_data

    # next 3 functions are Stage API Compliance
    def setup(self, m, i):
        pass

    def ispec(self):
        return FetchInput()

    def ospec(self):
        return FetchOutput()

    def elaborate(self, platform):
        """fetch FSM

        this FSM performs fetch of raw instruction data, partial-decodes
        it 32-bit at a time to detect SVP64 prefixes, and will optionally
        read a 2nd 32-bit quantity if that occurs.
        """
        m = super().elaborate(platform)

        dbg = self.dbg
        core = self.core
        pc = self.i.pc
        msr = self.i.msr
        svstate = self.svstate
        nia = self.nia
        fetch_pc_o_ready = self.p.o_ready
        fetch_pc_i_valid = self.p.i_valid
        fetch_insn_o_valid = self.n.o_valid
        fetch_insn_i_ready = self.n.i_ready

        comb = m.d.comb
        sync = m.d.sync
        pdecode2 = self.pdecode2
        cur_state = self.cur_state
        dec_opcode_o = pdecode2.dec.raw_opcode_in  # raw opcode

        # also note instruction fetch failed
        if hasattr(core, "icache"):
            fetch_failed = core.icache.i_out.fetch_failed
            flush_needed = True
        else:
            fetch_failed = Const(0, 1)
            flush_needed = False

        # set priv / virt mode on I-Cache, sigh
        if isinstance(self.imem, ICache):
            comb += self.imem.i_in.priv_mode.eq(~msr[MSR.PR])
            comb += self.imem.i_in.virt_mode.eq(msr[MSR.DR])

        with m.FSM(name='fetch_fsm'):

            # waiting (zzz)
            with m.State("IDLE"):
                with m.If(~dbg.stopping_o & ~fetch_failed):
                    comb += fetch_pc_o_ready.eq(1)
                with m.If(fetch_pc_i_valid & ~fetch_failed):
                    # instruction allowed to go: start by reading the PC
                    # capture the PC and also drop it into Insn Memory
                    # we have joined a pair of combinatorial memory
                    # lookups together.  this is Generally Bad.
                    comb += self.imem.a_pc_i.eq(pc)
                    comb += self.imem.a_i_valid.eq(1)
                    comb += self.imem.f_i_valid.eq(1)
                    sync += cur_state.pc.eq(pc)
                    sync += cur_state.svstate.eq(svstate)  # and svstate
                    sync += cur_state.msr.eq(msr)  # and msr

                    m.next = "INSN_READ"  # move to "wait for bus" phase

            # dummy pause to find out why simulation is not keeping up
            with m.State("INSN_READ"):
                if self.allow_overlap:
                    stopping = dbg.stopping_o
                else:
                    stopping = Const(0)
                with m.If(stopping):
                    # stopping: jump back to idle
                    m.next = "IDLE"
                with m.Else():
                    with m.If(self.imem.f_busy_o & ~fetch_failed):  # zzz...
                        # busy but not fetch failed: stay in wait-read
                        comb += self.imem.a_i_valid.eq(1)
                        comb += self.imem.f_i_valid.eq(1)
                    with m.Else():
                        # not busy (or fetch failed!): instruction fetched
                        # when fetch failed, the instruction gets ignored
                        # by the decoder
                        insn = get_insn(self.imem.f_instr_o, cur_state.pc)
                        # not SVP64 - 32-bit only
                        sync += nia.eq(cur_state.pc + 4)
                        sync += dec_opcode_o.eq(insn)
                            m.next = "INSN_READY"

            with m.State("INSN_READ2"):
                with m.If(self.imem.f_busy_o):  # zzz...
                    # busy: stay in wait-read
                    comb += self.imem.a_i_valid.eq(1)
                    comb += self.imem.f_i_valid.eq(1)
                with m.Else():
                    # not busy: instruction fetched
                    insn = get_insn(self.imem.f_instr_o, cur_state.pc+4)
                    sync += dec_opcode_o.eq(insn)
                    m.next = "INSN_READY"

            with m.State("INSN_READY"):
                # hand over the instruction, to be decoded
                comb += fetch_insn_o_valid.eq(1)
                with m.If(fetch_insn_i_ready):
                    m.next = "IDLE"

        # whatever was done above, over-ride it if core reset is held
        with m.If(self.core_rst):
            sync += nia.eq(0)

        return m


class TestIssuerInternalInOrder(TestIssuerBase):
    """TestIssuer - reads instructions from TestMemory and issues them

    efficiency and speed is not the main goal here: functional correctness
    and code clarity is.  optimisations (which almost 100% interfere with
    easy understanding) come later.
    """

    def issue_fsm(self, m, core, nia,
                  dbg, core_rst,
                  fetch_pc_o_ready, fetch_pc_i_valid,
                  fetch_insn_o_valid, fetch_insn_i_ready,
                  exec_insn_i_valid, exec_insn_o_ready,
                  exec_pc_o_valid, exec_pc_i_ready):
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
        dec_opcode_i = pdecode2.dec.raw_opcode_in  # raw opcode

        # note if an exception happened.  in a pipelined or OoO design
        # this needs to be accompanied by "shadowing" (or stalling)
        exc_happened = self.core.o.exc_happened
        # also note instruction fetch failed
        if hasattr(core, "icache"):
            fetch_failed = core.icache.i_out.fetch_failed
            flush_needed = True
            # set to fault in decoder
            # update (highest priority) instruction fault
            rising_fetch_failed = rising_edge(m, fetch_failed)
            with m.If(rising_fetch_failed):
                sync += pdecode2.instr_fault.eq(1)
        else:
            fetch_failed = Const(0, 1)
            flush_needed = False

        with m.FSM(name="issue_fsm"):

            # sync with the "fetch" phase which is reading the instruction
            # at this point, there is no instruction running, that
            # could inadvertently update the PC.
            with m.State("ISSUE_START"):
                # reset instruction fault
                sync += pdecode2.instr_fault.eq(0)
                # wait on "core stop" release, before next fetch
                # need to do this here, in case we are in a VL==0 loop
                with m.If(~dbg.core_stop_o & ~core_rst):
                    comb += fetch_pc_i_valid.eq(1)  # tell fetch to start
                    with m.If(fetch_pc_o_ready):   # fetch acknowledged us
                        m.next = "INSN_WAIT"
                with m.Else():
                    # tell core it's stopped, and acknowledge debug handshake
                    comb += dbg.core_stopped_i.eq(1)

            # wait for an instruction to arrive from Fetch
            with m.State("INSN_WAIT"):
                if self.allow_overlap:
                    stopping = dbg.stopping_o
                else:
                    stopping = Const(0)
                with m.If(stopping):
                    # stopping: jump back to idle
                    m.next = "ISSUE_START"
                    if flush_needed:
                        # request the icache to stop asserting "failed"
                        comb += core.icache.flush_in.eq(1)
                    # stop instruction fault
                    sync += pdecode2.instr_fault.eq(0)
                with m.Else():
                    comb += fetch_insn_i_ready.eq(1)
                    with m.If(fetch_insn_o_valid):
                        # loop into ISSUE_START if it's a SVP64 instruction
                        # and VL == 0.  this because VL==0 is a for-loop
                        # from 0 to 0 i.e. always, always a NOP.
                        m.next = "DECODE_SV"  # skip predication

            # after src/dst step have been updated, we are ready
            # to decode the instruction
            with m.State("DECODE_SV"):
                # decode the instruction
                with m.If(~fetch_failed):
                    sync += pdecode2.instr_fault.eq(0)
                sync += core.i.e.eq(pdecode2.e)
                sync += core.i.state.eq(cur_state)
                sync += core.i.raw_insn_i.eq(dec_opcode_i)
                sync += core.i.bigendian_i.eq(self.core_bigendian_i)
                # after decoding, reset any previous exception condition,
                # allowing it to be set again during the next execution
                sync += pdecode2.ldst_exc.eq(0)

                m.next = "INSN_EXECUTE"  # move to "execute"

            # handshake with execution FSM, move to "wait" once acknowledged
            with m.State("INSN_EXECUTE"):
                comb += exec_insn_i_valid.eq(1)  # trigger execute
                with m.If(exec_insn_o_ready):   # execute acknowledged us
                    m.next = "EXECUTE_WAIT"

            with m.State("EXECUTE_WAIT"):
                # wait on "core stop" release, at instruction end
                # need to do this here, in case we are in a VL>1 loop
                with m.If(~dbg.core_stop_o & ~core_rst):
                    comb += exec_pc_i_ready.eq(1)
                    # see https://bugs.libre-soc.org/show_bug.cgi?id=636
                    # the exception info needs to be blatted into
                    # pdecode.ldst_exc, and the instruction "re-run".
                    # when ldst_exc.happened is set, the PowerDecoder2
                    # reacts very differently: it re-writes the instruction
                    # with a "trap" (calls PowerDecoder2.trap()) which
                    # will *overwrite* whatever was requested and jump the
                    # PC to the exception address, as well as alter MSR.
                    # nothing else needs to be done other than to note
                    # the change of PC and MSR (and, later, SVSTATE)
                    with m.If(exc_happened):
                        mmu = core.fus.get_exc("mmu0")
                        ldst = core.fus.get_exc("ldst0")
                        if mmu is not None:
                            with m.If(fetch_failed):
                                # instruction fetch: exception is from MMU
                                # reset instr_fault (highest priority)
                                sync += pdecode2.ldst_exc.eq(mmu)
                                sync += pdecode2.instr_fault.eq(0)
                                if flush_needed:
                                    # request icache to stop asserting "failed"
                                    comb += core.icache.flush_in.eq(1)
                        with m.If(~fetch_failed):
                            # otherwise assume it was a LDST exception
                            sync += pdecode2.ldst_exc.eq(ldst)

                    with m.If(exec_pc_o_valid):

                        # return directly to Decode if Execute generated an
                        # exception.
                        with m.If(pdecode2.ldst_exc.happened):
                            m.next = "DECODE_SV"

                        # if MSR, PC or SVSTATE were changed by the previous
                        # instruction, go directly back to Fetch, without
                        # updating either MSR PC or SVSTATE
                        with m.Elif(self.msr_changed | self.pc_changed |
                                    self.sv_changed):
                            m.next = "ISSUE_START"

                        # returning to Execute? then, first update SRCSTEP
                        with m.Else():
                            # return to mask skip loop
                            m.next = "DECODE_SV"

                with m.Else():
                    comb += dbg.core_stopped_i.eq(1)
                    if flush_needed:
                        # request the icache to stop asserting "failed"
                        comb += core.icache.flush_in.eq(1)
                    # stop instruction fault
                    sync += pdecode2.instr_fault.eq(0)
                    if flush_needed:
                        # request the icache to stop asserting "failed"
                        comb += core.icache.flush_in.eq(1)
                    # stop instruction fault
                    sync += pdecode2.instr_fault.eq(0)

    def execute_fsm(self, m, core,
                    exec_insn_i_valid, exec_insn_o_ready,
                    exec_pc_o_valid, exec_pc_i_ready):
        """execute FSM

        execute FSM. this interacts with the "issue" FSM
        through exec_insn_ready/valid (incoming) and exec_pc_ready/valid
        (outgoing). SVP64 RM prefixes have already been set up by the
        "issue" phase, so execute is fairly straightforward.
        """

        comb = m.d.comb
        sync = m.d.sync
        pdecode2 = self.pdecode2

        # temporaries
        core_busy_o = core.n.o_data.busy_o  # core is busy
        core_ivalid_i = core.p.i_valid              # instruction is valid

        if hasattr(core, "icache"):
            fetch_failed = core.icache.i_out.fetch_failed
        else:
            fetch_failed = Const(0, 1)

        with m.FSM(name="exec_fsm"):

            # waiting for instruction bus (stays there until not busy)
            with m.State("INSN_START"):
                comb += exec_insn_o_ready.eq(1)
                with m.If(exec_insn_i_valid):
                    comb += core_ivalid_i.eq(1)  # instruction is valid/issued
                    sync += self.sv_changed.eq(0)
                    sync += self.pc_changed.eq(0)
                    sync += self.msr_changed.eq(0)
                    with m.If(core.p.o_ready):  # only move if accepted
                        m.next = "INSN_ACTIVE"  # move to "wait completion"

            # instruction started: must wait till it finishes
            with m.State("INSN_ACTIVE"):
                # note changes to MSR, PC and SVSTATE
                # XXX oops, really must monitor *all* State Regfile write
                # ports looking for changes!
                with m.If(self.state_nia.wen & (1 << StateRegs.SVSTATE)):
                    sync += self.sv_changed.eq(1)
                with m.If(self.state_nia.wen & (1 << StateRegs.MSR)):
                    sync += self.msr_changed.eq(1)
                with m.If(self.state_nia.wen & (1 << StateRegs.PC)):
                    sync += self.pc_changed.eq(1)
                with m.If(~core_busy_o):  # instruction done!
                    comb += exec_pc_o_valid.eq(1)
                    with m.If(exec_pc_i_ready):
                        # when finished, indicate "done".
                        # however, if there was an exception, the instruction
                        # is *not* yet done.  this is an implementation
                        # detail: we choose to implement exceptions by
                        # taking the exception information from the LDST
                        # unit, putting that *back* into the PowerDecoder2,
                        # and *re-running the entire instruction*.
                        # if we erroneously indicate "done" here, it is as if
                        # there were *TWO* instructions:
                        # 1) the failed LDST 2) a TRAP.
                        with m.If(~pdecode2.ldst_exc.happened &
                                  ~fetch_failed):
                            comb += self.insn_done.eq(1)
                        m.next = "INSN_START"  # back to fetch

    def elaborate(self, platform):
        m = super().elaborate(platform)
        # convenience
        comb, sync = m.d.comb, m.d.sync
        cur_state = self.cur_state
        pdecode2 = self.pdecode2
        dbg = self.dbg
        core = self.core

        # set up peripherals and core
        core_rst = self.core_rst

        # indicate to outside world if any FU is still executing
        comb += self.any_busy.eq(core.n.o_data.any_busy_o)  # any FU executing

        # address of the next instruction, in the absence of a branch
        # depends on the instruction size
        nia = Signal(64)

        # connect up debug signals
        comb += dbg.terminate_i.eq(core.o.core_terminate_o)

        # there are *THREE^WFOUR-if-SVP64-enabled* FSMs, fetch (32/64-bit)
        # issue, decode/execute, now joined by "Predicate fetch/calculate".
        # these are the handshake signals between each

        # fetch FSM can run as soon as the PC is valid
        fetch_pc_i_valid = Signal()  # Execute tells Fetch "start next read"
        fetch_pc_o_ready = Signal()  # Fetch Tells SVSTATE "proceed"

        # fetch FSM hands over the instruction to be decoded / issued
        fetch_insn_o_valid = Signal()
        fetch_insn_i_ready = Signal()

        # issue FSM delivers the instruction to the be executed
        exec_insn_i_valid = Signal()
        exec_insn_o_ready = Signal()

        # execute FSM, hands over the PC/SVSTATE back to the issue FSM
        exec_pc_o_valid = Signal()
        exec_pc_i_ready = Signal()

        # the FSMs here are perhaps unusual in that they detect conditions
        # then "hold" information, combinatorially, for the core
        # (as opposed to using sync - which would be on a clock's delay)
        # this includes the actual opcode, valid flags and so on.

        # Fetch, then predicate fetch, then Issue, then Execute.
        # Issue is where the VL for-loop # lives.  the ready/valid
        # signalling is used to communicate between the four.

        # set up Fetch FSM
        fetch = FetchFSM(self.allow_overlap,
                         self.imem, core_rst, pdecode2, cur_state,
                         dbg, core,
                         dbg.state.svstate, # combinatorially same
                         nia)
        m.submodules.fetch = fetch
        # connect up in/out data to existing Signals
        comb += fetch.p.i_data.pc.eq(dbg.state.pc)   # combinatorially same
        comb += fetch.p.i_data.msr.eq(dbg.state.msr) # combinatorially same
        # and the ready/valid signalling
        comb += fetch_pc_o_ready.eq(fetch.p.o_ready)
        comb += fetch.p.i_valid.eq(fetch_pc_i_valid)
        comb += fetch_insn_o_valid.eq(fetch.n.o_valid)
        comb += fetch.n.i_ready.eq(fetch_insn_i_ready)

        self.issue_fsm(m, core, nia,
                       dbg, core_rst,
                       fetch_pc_o_ready, fetch_pc_i_valid,
                       fetch_insn_o_valid, fetch_insn_i_ready,
                       exec_insn_i_valid, exec_insn_o_ready,
                       exec_pc_o_valid, exec_pc_i_ready)

        self.execute_fsm(m, core,
                         exec_insn_i_valid, exec_insn_o_ready,
                         exec_pc_o_valid, exec_pc_i_ready)

        return m


# XXX TODO: update this

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
