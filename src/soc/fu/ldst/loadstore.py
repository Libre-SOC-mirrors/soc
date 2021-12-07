"""LoadStore1 FSM.

based on microwatt loadstore1.vhdl, but conforming to PortInterface.
unlike loadstore1.vhdl this does *not* deal with actual Load/Store
ops: that job is handled by LDSTCompUnit, which talks to LoadStore1
by way of PortInterface.  PortInterface is where things need extending,
such as adding dcbz support, etc.

this module basically handles "pure" load / store operations, and
its first job is to ask the D-Cache for the data.  if that fails,
the second task (if virtual memory is enabled) is to ask the MMU
to perform a TLB, then to go *back* to the cache and ask again.

Links:

* https://bugs.libre-soc.org/show_bug.cgi?id=465

"""

from nmigen import (Elaboratable, Module, Signal, Shape, unsigned, Cat, Mux,
                    Record, Memory,
                    Const)
from nmutil.iocontrol import RecordObject
from nmutil.util import rising_edge, Display
from enum import Enum, unique

from soc.experiment.dcache import DCache
from soc.experiment.icache import ICache
from soc.experiment.pimem import PortInterfaceBase
from soc.experiment.mem_types import LoadStore1ToMMUType
from soc.experiment.mem_types import MMUToLoadStore1Type

from soc.minerva.wishbone import make_wb_layout
from soc.bus.sram import SRAM
from nmutil.util import Display


@unique
class State(Enum):
    IDLE = 0       # ready for instruction
    ACK_WAIT = 1   # waiting for ack from dcache
    MMU_LOOKUP = 2 # waiting for MMU to look up translation
    TLBIE_WAIT = 3 # waiting for MMU to finish doing a tlbie


# captures the LDSTRequest from the PortInterface, which "blips" most
# of this at us (pipeline-style).
class LDSTRequest(RecordObject):
    def __init__(self, name=None):
        RecordObject.__init__(self, name=name)

        self.load          = Signal()
        self.dcbz          = Signal()
        self.addr          = Signal(64)
        # self.store_data    = Signal(64) # this is already sync (on a delay)
        self.byte_sel      = Signal(8)
        self.nc            = Signal()              # non-cacheable access
        self.virt_mode     = Signal()
        self.priv_mode     = Signal()
        self.align_intr    = Signal()


# glue logic for microwatt mmu and dcache
class LoadStore1(PortInterfaceBase):
    def __init__(self, pspec):
        self.pspec = pspec
        self.disable_cache = (hasattr(pspec, "disable_cache") and
                              pspec.disable_cache == True)
        regwid = pspec.reg_wid
        addrwid = pspec.addr_wid

        super().__init__(regwid, addrwid)
        self.dcache = DCache()
        self.icache = ICache()
        # these names are from the perspective of here (LoadStore1)
        self.d_out  = self.dcache.d_in     # in to dcache is out for LoadStore
        self.d_in = self.dcache.d_out      # out from dcache is in for LoadStore
        self.i_out  = self.icache.i_in     # in to icache is out for LoadStore
        self.i_in = self.icache.i_out      # out from icache is in for LoadStore
        self.m_out  = LoadStore1ToMMUType() # out *to* MMU
        self.m_in = MMUToLoadStore1Type()   # in *from* MMU
        self.req = LDSTRequest(name="ldst_req")

        # TODO, convert dcache wb_in/wb_out to "standard" nmigen Wishbone bus
        self.dbus = Record(make_wb_layout(pspec))
        self.ibus = Record(make_wb_layout(pspec))

        # for creating a single clock blip to DCache
        self.d_valid = Signal()
        self.d_w_valid = Signal()
        self.d_validblip = Signal()

        # state info for LD/ST
        self.done          = Signal()
        self.done_delay    = Signal()
        # latch most of the input request
        self.load          = Signal()
        self.tlbie         = Signal()
        self.dcbz          = Signal()
        self.addr          = Signal(64)
        self.store_data    = Signal(64)
        self.load_data     = Signal(64)
        self.load_data_delay = Signal(64)
        self.byte_sel      = Signal(8)
        #self.xerc         : xer_common_t;
        #self.reserve       = Signal()
        #self.atomic        = Signal()
        #self.atomic_last   = Signal()
        #self.rc            = Signal()
        self.nc            = Signal()              # non-cacheable access
        self.virt_mode     = Signal()
        self.priv_mode     = Signal()
        self.state        = Signal(State)
        self.iside         = Signal() # request instruction-side load
        self.instr_fault   = Signal()
        self.align_intr    = Signal()
        self.busy          = Signal()
        self.wait_dcache   = Signal()
        self.wait_mmu      = Signal()
        #self.mode_32bit    = Signal()
        #self.intr_vec     : integer range 0 to 16#fff#;
        #self.nia           = Signal(64)
        #self.srr1          = Signal(16)
        # use these to set the dsisr or dar respectively
        self.mmu_set_spr    = Signal()
        self.mmu_set_dsisr  = Signal()
        self.mmu_set_dar    = Signal()
        self.sprval_in      = Signal(64)

        # ONLY access these read-only, do NOT attempt to change
        self.dsisr          = Signal(32)
        self.dar            = Signal(64)

    def set_wr_addr(self, m, addr, mask, misalign, msr_pr, is_dcbz):
        m.d.comb += self.req.load.eq(0) # store operation
        m.d.comb += self.req.byte_sel.eq(mask)
        m.d.comb += self.req.addr.eq(addr)
        m.d.comb += self.req.priv_mode.eq(~msr_pr) # not-problem  ==> priv
        m.d.comb += self.req.virt_mode.eq(msr_pr) # problem-state ==> virt
        m.d.comb += self.req.align_intr.eq(misalign)
        m.d.comb += self.req.dcbz.eq(is_dcbz)

        # m.d.comb += Display("set_wr_addr %i dcbz %i",addr,is_dcbz)

        # option to disable the cache entirely for write
        if self.disable_cache:
            m.d.comb += self.req.nc.eq(1)
        return None

    def set_rd_addr(self, m, addr, mask, misalign, msr_pr):
        m.d.comb += self.d_valid.eq(1)
        m.d.comb += self.req.load.eq(1) # load operation
        m.d.comb += self.req.byte_sel.eq(mask)
        m.d.comb += self.req.align_intr.eq(misalign)
        m.d.comb += self.req.addr.eq(addr)
        m.d.comb += self.req.priv_mode.eq(~msr_pr) # not-problem  ==> priv
        m.d.comb += self.req.virt_mode.eq(msr_pr) # problem-state ==> virt
        # BAD HACK! disable cacheing on LD when address is 0xCxxx_xxxx
        # this is for peripherals. same thing done in Microwatt loadstore1.vhdl
        with m.If(addr[28:] == Const(0xc, 4)):
            m.d.comb += self.req.nc.eq(1)
        # option to disable the cache entirely for read
        if self.disable_cache:
            m.d.comb += self.req.nc.eq(1)
        return None #FIXME return value

    def set_wr_data(self, m, data, wen):
        # do the "blip" on write data
        m.d.comb += self.d_valid.eq(1)
        # put data into comb which is picked up in main elaborate()
        m.d.comb += self.d_w_valid.eq(1)
        m.d.comb += self.store_data.eq(data)
        #m.d.sync += self.d_out.byte_sel.eq(wen) # this might not be needed
        st_ok = self.done # TODO indicates write data is valid
        return st_ok

    def get_rd_data(self, m):
        ld_ok = self.done_delay # indicates read data is valid
        data = self.load_data_delay   # actual read data
        return data, ld_ok

    def elaborate(self, platform):
        m = super().elaborate(platform)
        comb, sync = m.d.comb, m.d.sync

        # microwatt takes one more cycle before next operation can be issued
        sync += self.done_delay.eq(self.done)
        sync += self.load_data_delay.eq(self.load_data)

        # create dcache and icache module
        m.submodules.dcache = dcache = self.dcache
        m.submodules.icache = icache = self.icache

        # temp vars
        d_out, d_in, dbus = self.d_out, self.d_in, self.dbus
        i_out, i_in, ibus = self.i_out, self.i_in, self.ibus
        m_out, m_in = self.m_out, self.m_in
        exc = self.pi.exc_o
        exception = exc.happened
        mmureq = Signal()

        # copy of address, but gets over-ridden for OP_FETCH_FAILED
        maddr = Signal(64)
        m.d.comb += maddr.eq(self.addr)

        # create a blip (single pulse) on valid read/write request
        # this can be over-ridden in the FSM to get dcache to re-run
        # a request when MMU_LOOKUP completes.
        m.d.comb += self.d_validblip.eq(rising_edge(m, self.d_valid))
        ldst_r = LDSTRequest("ldst_r")
        comb += Display("MMUTEST: LoadStore1 d_in.error=%i",d_in.error)

        # fsm skeleton
        with m.Switch(self.state):
            with m.Case(State.IDLE):
                with m.If(self.d_validblip & ~exc.happened):
                    comb += self.busy.eq(1)
                    sync += self.state.eq(State.ACK_WAIT)
                    sync += ldst_r.eq(self.req) # copy of LDSTRequest on "blip"
#                   sync += Display("validblip self.req.virt_mode=%i",
#                   self.req.virt_mode)
                with m.Else():
                    sync += ldst_r.eq(0)

            # waiting for completion
            with m.Case(State.ACK_WAIT):
                comb += Display("MMUTEST: ACK_WAIT")
                comb += self.busy.eq(~exc.happened)

                with m.If(d_in.error):
                    # cache error is not necessarily "final", it could
                    # be that it was just a TLB miss
                    with m.If(d_in.cache_paradox):
                        comb += exception.eq(1)
                        sync += self.state.eq(State.IDLE)
                        sync += ldst_r.eq(0)
                        sync += Display("cache error -> update dsisr")
                        sync += self.dsisr[63 - 38].eq(~self.load)
                        # XXX there is no architected bit for this
                        # (probably should be a machine check in fact)
                        sync += self.dsisr[63 - 35].eq(d_in.cache_paradox)

                    with m.Else():
                        # Look up the translation for TLB miss
                        # and also for permission error and RC error
                        # in case the PTE has been updated.
                        comb += mmureq.eq(1)
                        sync += self.state.eq(State.MMU_LOOKUP)
                with m.If(d_in.valid):
                    m.d.comb += self.done.eq(~mmureq) # done if not doing MMU
                    with m.If(self.done):
                        sync += Display("ACK_WAIT, done %x", self.addr)
                    sync += self.state.eq(State.IDLE)
                    sync += ldst_r.eq(0)
                    with m.If(self.load):
                        m.d.comb += self.load_data.eq(d_in.data)

            # waiting here for the MMU TLB lookup to complete.
            # either re-try the dcache lookup or throw MMU exception
            with m.Case(State.MMU_LOOKUP):
                comb += self.busy.eq(~exception)
                with m.If(m_in.done):
                    with m.If(~self.instr_fault):
                        sync += Display("MMU_LOOKUP, done %x -> %x",
                                        self.addr, d_out.addr)
                        # retry the request now that the MMU has
                        # installed a TLB entry, if not exception raised
                        m.d.comb += self.d_out.valid.eq(~exception)
                        sync += self.state.eq(State.ACK_WAIT)
                        sync += ldst_r.eq(0)
                    with m.Else():
                        sync += self.state.eq(State.IDLE)

                with m.If(m_in.err):
                    # MMU RADIX exception thrown. XXX
                    # TODO: critical that the write here has to
                    # notify the MMU FSM of the change to dsisr
                    comb += exception.eq(1)
                    sync += Display("MMU RADIX exception thrown")
                    sync += Display("TODO: notify MMU of change to dsisr")
                    sync += self.dsisr[63 - 33].eq(m_in.invalid)
                    sync += self.dsisr[63 - 36].eq(m_in.perm_error) # noexec
                    sync += self.dsisr[63 - 38].eq(~self.load)
                    sync += self.dsisr[63 - 44].eq(m_in.badtree)
                    sync += self.dsisr[63 - 45].eq(m_in.rc_error)
                    sync += self.state.eq(State.IDLE)

            with m.Case(State.TLBIE_WAIT):
                pass

        # MMU FSM communicating a request to update DSISR or DAR (OP_MTSPR)
        with m.If(self.mmu_set_spr):
            with m.If(self.mmu_set_dsisr):
                sync += self.dsisr.eq(self.sprval_in)
            with m.If(self.mmu_set_dar):
                sync += self.dar.eq(self.sprval_in)

        # hmmm, alignment occurs in set_rd_addr/set_wr_addr, note exception
        with m.If(self.align_intr):
            comb += exc.happened.eq(1)
        # check for updating DAR
        with m.If(exception):
            sync += Display("exception %x", self.addr)
            # alignment error: store address in DAR
            with m.If(self.align_intr):
                sync += Display("alignment error: addr in DAR %x", self.addr)
                sync += self.dar.eq(self.addr)
            with m.Elif(~self.instr_fault):
                sync += Display("not instr fault, addr in DAR %x", self.addr)
                sync += self.dar.eq(self.addr)

        # when done or exception, return to idle state
        with m.If(self.done | exception):
            sync += self.state.eq(State.IDLE)
            comb += self.busy.eq(0)

        # happened, alignment, instr_fault, invalid.
        # note that all of these flow through - eventually to the TRAP
        # pipeline, via PowerDecoder2.
        comb += self.align_intr.eq(self.req.align_intr)
        comb += exc.invalid.eq(m_in.invalid)
        comb += exc.alignment.eq(self.align_intr)
        comb += exc.instr_fault.eq(self.instr_fault)
        # badtree, perm_error, rc_error, segment_fault
        comb += exc.badtree.eq(m_in.badtree)
        comb += exc.perm_error.eq(m_in.perm_error)
        comb += exc.rc_error.eq(m_in.rc_error)
        comb += exc.segment_fault.eq(m_in.segerr)

        # TODO, connect dcache wb_in/wb_out to "standard" nmigen Wishbone bus
        comb += dbus.adr.eq(dcache.bus.adr)
        comb += dbus.dat_w.eq(dcache.bus.dat_w)
        comb += dbus.sel.eq(dcache.bus.sel)
        comb += dbus.cyc.eq(dcache.bus.cyc)
        comb += dbus.stb.eq(dcache.bus.stb)
        comb += dbus.we.eq(dcache.bus.we)

        comb += dcache.bus.dat_r.eq(dbus.dat_r)
        comb += dcache.bus.ack.eq(dbus.ack)
        if hasattr(dbus, "stall"):
            comb += dcache.bus.stall.eq(dbus.stall)

        # update out d data when flag set
        with m.If(self.d_w_valid):
            m.d.sync += d_out.data.eq(self.store_data)
        #with m.Else():
        #    m.d.sync += d_out.data.eq(0)
        # unit test passes with that change

        # this must move into the FSM, conditionally noticing that
        # the "blip" comes from self.d_validblip.
        # task 1: look up in dcache
        # task 2: if dcache fails, look up in MMU.
        # do **NOT** confuse the two.
        with m.If(self.d_validblip):
            m.d.comb += self.d_out.valid.eq(~exc.happened)
            m.d.comb += d_out.load.eq(self.req.load)
            m.d.comb += d_out.byte_sel.eq(self.req.byte_sel)
            m.d.comb += self.addr.eq(self.req.addr)
            m.d.comb += d_out.nc.eq(self.req.nc)
            m.d.comb += d_out.priv_mode.eq(self.req.priv_mode)
            m.d.comb += d_out.virt_mode.eq(self.req.virt_mode)
            #m.d.comb += Display("validblip dcbz=%i addr=%x",
            #self.req.dcbz,self.req.addr)
            m.d.comb += d_out.dcbz.eq(self.req.dcbz)
        with m.Else():
            m.d.comb += d_out.load.eq(ldst_r.load)
            m.d.comb += d_out.byte_sel.eq(ldst_r.byte_sel)
            m.d.comb += self.addr.eq(ldst_r.addr)
            m.d.comb += d_out.nc.eq(ldst_r.nc)
            m.d.comb += d_out.priv_mode.eq(ldst_r.priv_mode)
            m.d.comb += d_out.virt_mode.eq(ldst_r.virt_mode)
            #m.d.comb += Display("no_validblip dcbz=%i addr=%x",
            #ldst_r.dcbz,ldst_r.addr)
            m.d.comb += d_out.dcbz.eq(ldst_r.dcbz)

        # XXX these should be possible to remove but for some reason
        # cannot be... yet. TODO, investigate
        m.d.comb += self.load_data.eq(d_in.data)
        m.d.comb += d_out.addr.eq(self.addr)

        # Update outputs to MMU
        m.d.comb += m_out.valid.eq(mmureq)
        m.d.comb += m_out.iside.eq(self.iside)
        m.d.comb += m_out.load.eq(ldst_r.load)
        # m_out.priv <= r.priv_mode; TODO
        m.d.comb += m_out.tlbie.eq(self.tlbie)
        # m_out.mtspr <= mmu_mtspr; # TODO
        # m_out.sprn <= sprn; # TODO
        m.d.comb += m_out.addr.eq(maddr)
        # m_out.slbia <= l_in.insn(7); # TODO: no idea what this is
        # m_out.rs <= l_in.data; # nope, probably not needed, TODO investigate

        return m

    def ports(self):
        yield from super().ports()
        # TODO: memory ports


class TestSRAMLoadStore1(LoadStore1):
    def __init__(self, pspec):
        super().__init__(pspec)
        pspec = self.pspec
        # small 32-entry Memory
        if (hasattr(pspec, "dmem_test_depth") and
                isinstance(pspec.dmem_test_depth, int)):
            depth = pspec.dmem_test_depth
        else:
            depth = 32
        print("TestSRAMBareLoadStoreUnit depth", depth)

        self.mem = Memory(width=pspec.reg_wid, depth=depth)

    def elaborate(self, platform):
        m = super().elaborate(platform)
        comb = m.d.comb
        m.submodules.sram = sram = SRAM(memory=self.mem, granularity=8,
                                        features={'cti', 'bte', 'err'})
        dbus = self.dbus

        # directly connect the wishbone bus of LoadStoreUnitInterface to SRAM
        # note: SRAM is a target (slave), dbus is initiator (master)
        fanouts = ['dat_w', 'sel', 'cyc', 'stb', 'we', 'cti', 'bte']
        fanins = ['dat_r', 'ack', 'err']
        for fanout in fanouts:
            print("fanout", fanout, getattr(sram.bus, fanout).shape(),
                  getattr(dbus, fanout).shape())
            comb += getattr(sram.bus, fanout).eq(getattr(dbus, fanout))
            comb += getattr(sram.bus, fanout).eq(getattr(dbus, fanout))
        for fanin in fanins:
            comb += getattr(dbus, fanin).eq(getattr(sram.bus, fanin))
        # connect address
        comb += sram.bus.adr.eq(dbus.adr)

        return m

