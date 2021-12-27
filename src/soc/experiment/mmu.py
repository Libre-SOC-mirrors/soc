# MMU
#
# License for original copyright mmu.vhdl by microwatt authors: CC4
# License for copyrighted modifications made in mmu.py: LGPLv3+
#
# This derivative work although includes CC4 licensed material is
# covered by the LGPLv3+

"""MMU

based on Anton Blanchard microwatt mmu.vhdl

"""
from enum import Enum, unique
from nmigen import (C, Module, Signal, Elaboratable, Mux, Cat, Repl, Signal)
from nmigen.cli import main
from nmigen.cli import rtlil
from nmutil.iocontrol import RecordObject
from nmutil.byterev import byte_reverse
from nmutil.mask import Mask, masked
from nmutil.util import Display

# NOTE: to use cxxsim, export NMIGEN_SIM_MODE=cxxsim from the shell
# Also, check out the cxxsim nmigen branch, and latest yosys from git
from nmutil.sim_tmp_alternative import Simulator, Settle

from nmutil.util import wrap

from soc.experiment.mem_types import (LoadStore1ToMMUType,
                                 MMUToLoadStore1Type,
                                 MMUToDCacheType,
                                 DCacheToMMUType,
                                 MMUToICacheType)

# Radix Tree Page Directory Entry Record, TODO put this somewhere sensible
# v3.0C Book III p1015-1016 section 6.7.10.1
class RTPDE(RecordObject):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.nls   = Signal(5)  # Nextded Access Auth bits 59:63 LSB0 0:4
        self.rs1   = Signal(3)  # Reserved            bits 56:58 LSB0 5:7
        self.nlb   = Signal(52) # Next Level Base     bit  4:55  LSB0 8:59
        self.rs2   = Signal(2)  # Reserved            bit  2:3   LSB0 60:61
        self.leaf  = Signal(1)  # leaf                bit  1     LSB0 62
        self.valid = Signal(1)  # valid               bit  0     LSB0 63


# Radix Tree Page Table Entry Record, TODO put this somewhere sensible
# v3.0C Book III p1016 section 6.7.10.2
class RTPTE(RecordObject):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.eaa   = Signal(4)  # Encoded Access Auth bits 60:63 LSB0 0:3
        self.att   = Signal(2)  # Attributes          bits 58:59 LSB0 4:5
        self.rs1   = Signal(1)  # Reserved            bit  57    LSB0 6
        self.c     = Signal(1)  # Change              bit  56    LSB0 7
        self.r     = Signal(1)  # Reference           bit  55    LSB0 8
        self.sw    = Signal(3)  # SW bits 1:3         bits 52:54 LSB0 9:11
        self.rpn   = Signal(45) # Real Page Number    bits 7:51  LSB0 12:56
        self.rs2   = Signal(4)  # Reserved            bit  3:6   LSB0 57-60
        self.sw0   = Signal(1)  # SW bit 0            bit  2     LSB0 61
        self.leaf  = Signal(1)  # leaf                bit  1     LSB0 62
        self.valid = Signal(1)  # valid               bit  0     LSB0 63

# and these... which of course are turned round to LSB0 order.
# TODO: sigh. use botchify and put them in openpower.consts
EAA_PRIV = 3 # bit 0 (in MSB0) set ==> problem-state banned (priv=1 only)
EAA_RD   = 2 # bit 1 (in MSB0) set ==> loads are permitted
EAA_WR   = 1 # bit 2 (in MSB0) set ==> load and stores permitted
EAA_EXE  = 0 # bit 3 (in MSB0) set ==> execute permitted

# for debugging
display_invalid = True

@unique
class State(Enum):
    IDLE = 0            # zero is default on reset for r.state
    DO_TLBIE = 1
    TLB_WAIT = 2
    PROC_TBL_READ = 3
    PROC_TBL_WAIT = 4
    SEGMENT_CHECK = 5
    RADIX_LOOKUP = 6
    RADIX_READ_WAIT = 7
    RADIX_LOAD_TLB = 8
    RADIX_FINISH = 9


# Process Table Record - near-identical to Page Table Record (same format)
# v3.0C Book III Section 6.7.6.2 p1004
class PRTBL(RecordObject):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.rpds  = Signal(5)  # Root Page Directory Size  59:63 LSB0 0:4
        self.rts2  = Signal(3)  # Radix Tree Size part 2    56:58 LSB0 5:7
        self.rpdb  = Signal(52) # Root Page Directory Base  4:55  LSB0 8:59
        self.rsv2  = Signal(1)  # reserved                  3     LSB0 60
        self.rts1  = Signal(2)  # Radix Tree Size part 1    1:2   LSB0 61:62
        self.rsv1  = Signal(1)  # reserved                  0     LSB0 63


class RegStage(RecordObject):
    def __init__(self, name=None):
        super().__init__(name=name)
        # latched request from loadstore1
        self.valid = Signal()
        self.iside = Signal()
        self.store = Signal()
        self.priv = Signal()
        self.addr = Signal(64)
        self.inval_all = Signal()

        # config SPRs
        self.prtbl = Signal(64)
        self.pid = Signal(32)

        # internal state
        self.state = Signal(State) # resets to IDLE
        self.done = Signal()
        self.err = Signal()

        # there are 4 quadrants (0-3): here we only support 2 (pt0 and pt3)
        # these are bits 62-63 of any given address.
        # except in segment_check, bit 62 is ignored
        # Quadrant Select can be seen in v3.0C 6.7.10 p1015 book III figure 36
        # and is further described in 6.7.11.3 p1019
        self.pgtbl0 = Signal(64)
        self.pt0_valid = Signal()
        self.pgtbl3 = Signal(64)
        self.pt3_valid = Signal()

        self.shift = Signal(6)
        self.mask_size = Signal(5)
        self.pgbase = Signal(56)
        self.pde = Signal(64)
        self.invalid = Signal()
        self.badtree = Signal()
        self.segerror = Signal()
        self.perm_err = Signal()
        self.rc_error = Signal()


# Page Table Record - note that HR bit is treated as part of rts below
# (near-identical to Process Table Record - same format)
# v3.0C Book III Section 6.7.6.1 p1003
class PGTBL(RecordObject):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.rpds  = Signal(5)  # Root Page Directory Size  59:63 LSB0 0:4
        self.rts2  = Signal(3)  # Radix Tree Size part 2    56:58 LSB0 5:7
        self.rpdb  = Signal(52) # Root Page Directory Base  4:55  LSB0 8:59
        self.s     = Signal(1)  # Host Secure               3     LSB0 60
        self.rts1  = Signal(2)  # Radix Tree Size part 1    1:2   LSB0 61:62
        self.hr    = Signal(1)  # Host Radix                0     LSB0 63


class MMU(Elaboratable):
    """Radix MMU

    Supports 4-level trees as in arch 3.0B, but not the
    two-step translation for guests under a hypervisor
    (i.e. there is no gRA -> hRA translation).
    """
    def __init__(self):
        self.l_in  = LoadStore1ToMMUType("l_in")
        self.l_out = MMUToLoadStore1Type("l_out")
        self.d_out = MMUToDCacheType("d_out")
        self.d_in  = DCacheToMMUType("d_in")
        self.i_out = MMUToICacheType("i_out")

    def radix_tree_idle(self, m, l_in, r, v):
        """radix_tree_idle - the main decision-point.  valid actions include:
        * LDST incoming TLBIE request (invalidate TLB entry)
        * LDST incoming RADIX walk request
        * set either PRTBL or PID SPRs (which then fires a TLB invalidate)
        """
        comb = m.d.comb
        sync = m.d.sync

        pt_valid = Signal()
        pgtbl = PGTBL("pgtbl")
        rts = Signal(6)
        mbits = Signal(6, name="mbits_idle")

        with m.If(l_in.addr[63]): # quadrant 3
            comb += pgtbl.eq(r.pgtbl3)
            comb += pt_valid.eq(r.pt3_valid)
        with m.Else():
            comb += pgtbl.eq(r.pgtbl0)
            comb += pt_valid.eq(r.pt0_valid)

        # rts == radix tree size, number of address bits
        # being translated.  takes bits 5:7 and 61:62
        comb += rts.eq(Cat(pgtbl.rts2, pgtbl.rts1, C(0)))

        # mbits == number of address bits to index top
        # level of tree.  takes bits 0:4
        comb += mbits.eq(pgtbl.rpds)

        # set v.shift to rts so that we can use finalmask
        # for the segment check.
        # note: rpdb (52 bits long) is truncated to 48 bits
        comb += v.shift.eq(rts)
        comb += v.mask_size.eq(mbits[0:5])

        # create the page base from root page directory base (48 bits with 8 0s)
        comb += v.pgbase.eq(Cat(C(0, 8), pgtbl.rpdb[:48])) # bits 8:55

        # request either TLB invalidate
        # or start a RADIX walk

        with m.If(l_in.valid):
            comb += v.addr.eq(l_in.addr)
            comb += v.iside.eq(l_in.iside)
            comb += v.store.eq(~(l_in.load | l_in.iside))
            comb += v.priv.eq(l_in.priv)

            sync += Display("state %d l_in.valid addr %x iside %d store %d "
                            "rpdb %x rts %d mbits %d pt_valid %d",
                            v.state, v.addr, v.iside, v.store,
                            pgtbl.rpdb, rts, mbits, pt_valid)

            with m.If(l_in.tlbie):
                # Invalidate all iTLB/dTLB entries for
                # tlbie with RB[IS] != 0 or RB[AP] != 0,
                # or for slbia
                comb += v.inval_all.eq(l_in.slbia
                                       | l_in.addr[11]
                                       | l_in.addr[10]
                                       | l_in.addr[7]
                                       | l_in.addr[6]
                                       | l_in.addr[5]
                                      )
                # The RIC field of the tlbie instruction
                # comes across on the sprn bus as bits 2--3.
                # RIC=2 flushes process table caches.
                with m.If(l_in.sprn[3]):
                    comb += v.pt0_valid.eq(0)
                    comb += v.pt3_valid.eq(0)
                comb += v.state.eq(State.DO_TLBIE)
            with m.Else():
                comb += v.valid.eq(1)
                with m.If(~pt_valid):
                    # need to fetch process table entry
                    # set v.shift so we can use finalmask
                    # for generating the process table
                    # entry address
                    prtbl = PRTBL("prtbl")
                    comb += prtbl.eq(r.prtbl)
                    comb += v.shift.eq(prtbl.rpds)
                    comb += v.state.eq(State.PROC_TBL_READ)

                with m.Elif(mbits == 0):
                    # Use RPDS = 0 to disable radix tree walks
                    comb += v.state.eq(State.RADIX_FINISH)
                    comb += v.invalid.eq(1)
                    if(display_invalid):
                        sync += Display("MMUBUG: Use RPDS = 0 to disable"
                                        " radix tree walks")
                with m.Else():
                    comb += v.state.eq(State.SEGMENT_CHECK)

        # set either PID or PRTBL SPRs
        # (then invalidate TLBs)

        with m.If(l_in.mtspr):
            # Move to PID needs to invalidate L1 TLBs
            # and cached pgtbl0 value.
            # Move to PRTBL does that plus invalidating the cached
            # pgtbl3 value as well.
            with m.If(~l_in.sprn[9]):
                comb += v.pid.eq(l_in.rs[0:32])
            with m.Else():
                comb += v.prtbl.eq(l_in.rs)
                comb += v.pt3_valid.eq(0)

            comb += v.pt0_valid.eq(0)
            comb += v.inval_all.eq(1)
            comb += v.state.eq(State.DO_TLBIE)

    def proc_tbl_wait(self, m, v, r, data):
        comb = m.d.comb
        sync = m.d.sync
        rts = Signal(6)
        mbits = Signal(6, name="mbits_tbl_wait")
        prtbl = PRTBL("prtblw")
        comb += prtbl.eq(data)

        with m.If(r.addr[63]): # top bit of quadrant selects pt3
            comb += v.pgtbl3.eq(prtbl)
            comb += v.pt3_valid.eq(1)
        with m.Else():
            comb += v.pgtbl0.eq(prtbl)
            comb += v.pt0_valid.eq(1)

        # rts == radix tree size, # address bits being translated
        comb += rts.eq(Cat(prtbl.rts2, prtbl.rts1, C(0)))

        # mbits == # address bits to index top level of tree
        comb += mbits.eq(prtbl.rpds[0:5])

        # set v.shift to rts so that we can use finalmask for the segment check
        comb += v.shift.eq(rts)
        comb += v.mask_size.eq(mbits[0:5])

        # create the page base from root page directory base (48 bits with 8 0s)
        comb += v.pgbase.eq(Cat(C(0, 8), prtbl.rpdb[:48])) # bits 8:55

        with m.If(mbits):
            comb += v.state.eq(State.SEGMENT_CHECK)
            sync += Display("PROC TBL %d data %x rts1 %x rts2 %x rts %d "
                            "rpdb %x mbits %d pgbase %x "
                            " pt0_valid %d, pt3_valid %d",
                            v.state, data, prtbl.rts1, prtbl.rts2, rts,
                            prtbl.rpdb, mbits, v.pgbase,
                            v.pt0_valid, v.pt3_valid)
        with m.Else():
            comb += v.state.eq(State.RADIX_FINISH)
            comb += v.invalid.eq(1)
            if (display_invalid): m.d.sync += Display("MMU: mbits is invalid")

    def radix_read_wait(self, m, v, r, d_in, data):
        comb = m.d.comb
        sync = m.d.sync

        rpte = RTPTE(name="radix_rpte") # page-table (leaf) entry
        rpde = RTPDE(name="radix_rpde") # page-directory (non-leaf) entry

        perm_ok = Signal()
        rc_ok = Signal()
        mbits = Signal(6, name="mbits_read_wait")
        valid = rpte.valid
        eaa = rpte.eaa
        leaf = rpte.leaf
        badtree = Signal()

        sync += Display("RDW %016x done %d "
                        "perm %d rc %d mbits %d shf %d "
                        "valid %d leaf %d bad %d",
                        data, d_in.done, perm_ok, rc_ok,
                        mbits, r.shift, valid, leaf, badtree)

        # set pde and interpret as Radix Tree Page Table Entry (leaf=1 case)
        comb += v.pde.eq(data)
        comb += rpte.eq(data)
        comb += rpde.eq(data)

        with m.If(valid):
            # valid & leaf: RADIX Page-Table Entry
            with m.If(leaf):
                # check permissions and RC bits
                with m.If(r.priv | ~eaa[EAA_PRIV]):
                    with m.If(r.iside): # instruction-side request
                        # no IAMR, so no KUEP support for now
                        # deny execute permission if cache inhibited
                        comb += perm_ok.eq(eaa[EAA_EXE] & ~rpte.att[1])
                    with m.Else():
                        # Load/Store (read/write)
                        comb += perm_ok.eq(eaa[EAA_WR] |
                                          (eaa[EAA_RD] & ~r.store))
                comb += rc_ok.eq(rpte.r & (rpte.c | ~r.store))

                # permissions / rc ok, load TLB, otherwise report error
                with m.If(perm_ok & rc_ok):
                    comb += v.state.eq(State.RADIX_LOAD_TLB)
                    sync += Display("RADIX LEAF data %x att %x eaa %x "
                                    "R %d C %d "
                                    "shift %d pgbase %x ",
                                    data, rpte.att, eaa,
                                    rpte.r, rpte.c,
                                    v.shift, v.pgbase
                                    )
                with m.Else():
                    comb += v.state.eq(State.RADIX_FINISH)
                    comb += v.perm_err.eq(~perm_ok)
                    # permission error takes precedence over RC error
                    comb += v.rc_error.eq(perm_ok)

            # valid & !leaf: RADIX Page-Directory Entry
            with m.Else():
                comb += mbits.eq(rpde.nls) # 5 bits NLS into 6-bit-long mbits
                comb += badtree.eq((mbits < 5) |
                                   (mbits > 16) |
                                   (mbits > r.shift))
                with m.If(badtree):
                    comb += v.state.eq(State.RADIX_FINISH)
                    comb += v.badtree.eq(1)
                with m.Else():
                    comb += v.shift.eq(r.shift - mbits)
                    comb += v.mask_size.eq(mbits)
                    # pagebase is first 48 bits of NLB, shifted up 1 byte
                    comb += v.pgbase.eq(Cat(C(0, 8), rpde.nlb[:48]))
                    comb += v.state.eq(State.RADIX_LOOKUP)

        with m.Else():
            # non-present PTE, generate a DSI
            comb += v.state.eq(State.RADIX_FINISH)
            comb += v.invalid.eq(1)
            if (display_invalid):
                sync += Display("MMU: non-present PTE, generate a DSI")

    def segment_check(self, m, v, r, data, finalmask):
        """segment_check: checks validity of the request before doing a
        RADIX lookup. reports either segment error or bad tree if not ok
        """
        comb = m.d.comb

        mbits = Signal(6, name="mbits_check")
        nonzero = Signal()
        comb += mbits.eq(r.mask_size)
        comb += v.shift.eq(r.shift + (31 - 12) - mbits)
        comb += nonzero.eq((r.addr[31:62] & ~finalmask[0:31]).bool())
        with m.If((r.addr[63] != r.addr[62]) # pt3 == 0b11 and pt1 == 0b00
                  | nonzero):
            comb += v.state.eq(State.RADIX_FINISH)
            comb += v.segerror.eq(1)
        with m.Elif((mbits < 5) | (mbits > 16) |
                    (mbits > (r.shift + (31-12)))):
            comb += v.state.eq(State.RADIX_FINISH)
            comb += v.badtree.eq(1)
        with m.Else():
            comb += v.state.eq(State.RADIX_LOOKUP)

    def mmu_0(self, m, r, rin, l_in, l_out, d_out, addrsh, mask):
        comb = m.d.comb
        sync = m.d.sync

        # Multiplex internal SPR values back to loadstore1,
        # selected by l_in.sprn.
        with m.If(l_in.sprn[9]):
            comb += l_out.sprval.eq(r.prtbl)
        with m.Else():
            comb += l_out.sprval.eq(r.pid)

        with m.If(rin.valid):
            sync += Display("MMU got tlb miss for %x", rin.addr)

        with m.If(l_out.done):
            sync += Display("MMU completing op without error")

        with m.If(l_out.err):
            sync += Display("MMU completing op with err invalid="
                            "%d badtree=%d", l_out.invalid, l_out.badtree)

        with m.If(rin.state == State.RADIX_LOOKUP):
            sync += Display ("radix lookup shift=%x msize=%x",
                            rin.shift, mask)

        with m.If(r.state == State.RADIX_LOOKUP):
            sync += Display(f"send load addr=%x addrsh=%x mask=%x",
                            d_out.addr, addrsh, mask)

        # update the internal register
        sync += r.eq(rin)

    def elaborate(self, platform):
        m = Module()

        comb = m.d.comb
        sync = m.d.sync

        addrsh = Signal(16)
        mask = Signal(16)
        finalmask = Signal(44)

        self.rin = rin = RegStage("r_in")
        r = RegStage("r")

        # get access to prtbl and pid for debug / testing purposes ONLY
        # (actually, not needed, because setup_regs() triggers mmu direct)
        # self._prtbl = r.prtbl
        # self._pid = r.pid

        l_in  = self.l_in
        l_out = self.l_out
        d_out = self.d_out
        d_in  = self.d_in
        i_out = self.i_out

        self.mmu_0(m, r, rin, l_in, l_out, d_out, addrsh, mask)

        v = RegStage("v")
        dcreq = Signal()
        tlb_load = Signal()
        itlb_load = Signal()
        tlbie_req = Signal()
        prtbl_rd = Signal()
        effpid = Signal(32)
        prtb_adr = Signal(64)
        pgtb_adr = Signal(64)
        pte = Signal(64)
        tlb_data = Signal(64)
        addr = Signal(64)

        comb += v.eq(r)
        comb += v.valid.eq(0)
        comb += v.done.eq(0)
        comb += v.err.eq(0)
        comb += v.invalid.eq(0)
        comb += v.badtree.eq(0)
        comb += v.segerror.eq(0)
        comb += v.perm_err.eq(0)
        comb += v.rc_error.eq(0)
        comb += v.inval_all.eq(0)

        # Radix tree data structures in memory are
        # big-endian, so we need to byte-swap them
        data = byte_reverse(m, "data", d_in.data, 8)

        # generate mask for extracting address fields for PTE addr generation
        m.submodules.pte_mask = pte_mask = Mask(16-5)
        pte_mask.mask.name = "pte_mask"
        comb += pte_mask.shift.eq(r.mask_size - 5)
        comb += mask.eq(Cat(C(0x1f, 5), pte_mask.mask))

        # generate mask for extracting address bits to go in
        # TLB entry in order to support pages > 4kB
        m.submodules.tlb_mask = tlb_mask = Mask(44)
        tlb_mask.mask.name = "tlb_mask"
        comb += tlb_mask.shift.eq(r.shift)
        comb += finalmask.eq(tlb_mask.mask)

        # Shift address bits 61--12 right by 0--47 bits and
        # supply the least significant 16 bits of the result.
        comb += addrsh.eq(r.addr[12:62] << r.shift)

        with m.If(r.state != State.IDLE):
            sync += Display("MMU state %d %016x", r.state, data)
            sync += Display("addrsh %x r.shift %d r.addr[12:62] %x",
                        addrsh, r.shift, r.addr[12:62])

        ##########
        # Main FSM
        ##########

        with m.Switch(r.state):
            with m.Case(State.IDLE):
                self.radix_tree_idle(m, l_in, r, v)

            with m.Case(State.DO_TLBIE):
                comb += dcreq.eq(1)
                comb += tlbie_req.eq(1)
                comb += v.state.eq(State.TLB_WAIT)

            with m.Case(State.TLB_WAIT):
                with m.If(d_in.done):
                    comb += v.state.eq(State.RADIX_FINISH)

            with m.Case(State.PROC_TBL_READ):
                sync += Display("   TBL_READ %016x", prtb_adr)
                comb += dcreq.eq(1)
                comb += prtbl_rd.eq(1)
                comb += v.state.eq(State.PROC_TBL_WAIT)

            with m.Case(State.PROC_TBL_WAIT):
                with m.If(d_in.done):
                    self.proc_tbl_wait(m, v, r, data)

                with m.If(d_in.err):
                    comb += v.state.eq(State.RADIX_FINISH)
                    comb += v.badtree.eq(1)

            with m.Case(State.SEGMENT_CHECK):
                self.segment_check(m, v, r, data, finalmask)

            with m.Case(State.RADIX_LOOKUP):
                sync += Display("   RADIX_LOOKUP")
                comb += dcreq.eq(1)
                comb += v.state.eq(State.RADIX_READ_WAIT)

            with m.Case(State.RADIX_READ_WAIT):
                sync += Display("   READ_WAIT")
                with m.If(d_in.done):
                    self.radix_read_wait(m, v, r, d_in, data)
                with m.If(d_in.err):
                    comb += v.state.eq(State.RADIX_FINISH)
                    comb += v.badtree.eq(1)

            with m.Case(State.RADIX_LOAD_TLB):
                comb +=  tlb_load.eq(1)
                with m.If(~r.iside):
                    comb += dcreq.eq(1)
                    comb += v.state.eq(State.TLB_WAIT)
                with m.Else():
                    comb += itlb_load.eq(1)
                    comb += v.state.eq(State.IDLE)

            with m.Case(State.RADIX_FINISH):
                sync += Display("   RADIX_FINISH")
                comb += v.state.eq(State.IDLE)

        # check and report either error or done.
        with m.If((v.state == State.RADIX_FINISH) |
                 ((v.state == State.RADIX_LOAD_TLB) & r.iside)):
            comb += v.err.eq(v.invalid | v.badtree | v.segerror
                             | v.perm_err | v.rc_error)
            comb += v.done.eq(~v.err)

        # PID is only valid if MSB of address is zero, top 2 bits are Quadrant
        with m.If(~r.addr[63]): # quadrant 0 (pt0)
            comb += effpid.eq(r.pid)

        # calculate Process Table Address
        pr24 = Signal(24, reset_less=True)
        prtbla = PRTBL("prtbla")
        comb += prtbla.eq(r.prtbl)
        rpdb = prtbla.rpdb
        comb += pr24.eq(masked(rpdb[4:28], effpid[8:32], finalmask))
        comb += prtb_adr.eq(Cat(C(0, 4), effpid[0:8], pr24, rpdb[28:48]))

        # calculate Page Table Address
        pg16 = Signal(16, reset_less=True)
        comb += pg16.eq(masked(r.pgbase[3:19], addrsh, mask))
        comb += pgtb_adr.eq(Cat(C(0, 3), pg16, r.pgbase[19:56]))

        # calculate Page Table Entry from Real Page Number (leaf=1, RTPTE)
        rpte = RTPTE(name="rpte")
        comb += rpte.eq(r.pde)
        pd44 = Signal(44, reset_less=True)
        comb += pd44.eq(masked(rpte.rpn, r.addr[12:56], finalmask))
        comb += pte.eq(Cat(r.pde[0:12], pd44))

        # update registers
        comb += rin.eq(v)

        # drive outputs
        with m.If(tlbie_req):
            comb += addr.eq(r.addr)
        with m.Elif(tlb_load):
            comb += addr.eq(Cat(C(0, 12), r.addr[12:64]))
            comb += tlb_data.eq(pte)
        with m.Elif(prtbl_rd):
            comb += addr.eq(prtb_adr)
        with m.Else():
            comb += addr.eq(pgtb_adr)
            sync += Display(f"pagetable pg16=%x addrsh %x mask %x pgbase=%x "
                            "pgbase[19:56]=%x",
                            pg16, addrsh, mask, r.pgbase, r.pgbase[19:56])

        # connect to other interfaces: LDST, D-Cache, I-Cache
        comb += l_out.done.eq(r.done)
        comb += l_out.err.eq(r.err)
        comb += l_out.invalid.eq(r.invalid)
        comb += l_out.badtree.eq(r.badtree)
        comb += l_out.segerr.eq(r.segerror)
        comb += l_out.perm_error.eq(r.perm_err)
        comb += l_out.rc_error.eq(r.rc_error)

        comb += d_out.valid.eq(dcreq)
        comb += d_out.tlbie.eq(tlbie_req)
        comb += d_out.doall.eq(r.inval_all)
        comb += d_out.tlbld.eq(tlb_load)
        comb += d_out.addr.eq(addr)
        comb += d_out.pte.eq(tlb_data)

        comb += i_out.tlbld.eq(itlb_load)
        comb += i_out.tlbie.eq(tlbie_req)
        comb += i_out.doall.eq(r.inval_all)
        comb += i_out.addr.eq(addr)
        comb += i_out.pte.eq(tlb_data)

        return m

stop = False

def dcache_get(dut):
    """simulator process for getting memory load requests
    """

    global stop

    def b(x):
        return int.from_bytes(x.to_bytes(8, byteorder='little'),
                              byteorder='big', signed=False)

    mem = {0x0: 0x000000, # to get mtspr prtbl working

           0x10000:    # PARTITION_TABLE_2
                       # HR=1 RTS1=0x2 PRTB=0x300 RTS2=0x5 PRTS=0xb
           b(0xc0000000000030ad),

           0x30000:     # RADIX_ROOT_PTE
                        # V = 1 L = 0 NLB = 0x400 NLS = 9
           b(0x8000000000040009),

           0x40000:     # RADIX_SECOND_LEVEL
                        # 	   V = 1 L = 1 SW = 0 RPN = 0
	                    # R = 1 C = 1 ATT = 0 EAA 0x7
           b(0xc000000000000187),

#
#   slightly different from radix_walk_example.txt: address in microwatt
#   has the top bit set to indicate hypervisor.  here, Quadrant 3's
#   process table entry is put instead into Quadrant 0.  the entry
#   PROCESS_TABLE_3 should, strictly speaking, be at 0x1000010

#          0x1000000:   # PROCESS_TABLE_3 (pt0_valid)
#                       # RTS1 = 0x2 RPDB = 0x300 RTS2 = 0x5 RPDS = 12
#           b(0x40000000000300ac),

          0x1000000:   # PROCESS_TABLE_3 (pt3_valid)
                       # RTS1 = 0x2 RPDB = 0x300 RTS2 = 0x5 RPDS = 13
           b(0x40000000000300ad),
          }

    # microwatt mmu.bin first part of test 2.
    # PRTBL must be set to 0x12000, PID to 1
    mem = {
             0x0: 0x000000, # to get mtspr prtbl working
             0x13920: 0x86810000000000c0, # leaf, supposed to be at 0x13920
             0x10000: 0x0930010000000080, # directory node
             0x12010: 0x0a00010000000000, # page table
             0x124000: 0x0000000badc0ffee,  # memory to be looked up
            }

    # microwatt mmu.bin first part of test 4.
    # PRTBL must be set to 0x12000, PID to 1
    mem = {
             0x0: 0x000000, # to get mtspr prtbl working
             0x13858: 0x86a10000000000c0, # leaf node
             0x10000: 0x0930010000000080, # directory node
             0x12010: 0x0a00010000000000, # page table
    }

    while not stop:
        while True: # wait for dc_valid
            if stop:
                return
            dc_valid = yield (dut.d_out.valid)
            tlbld = yield (dut.d_out.tlbld)
            if dc_valid:
                break
            yield
        addr = yield dut.d_out.addr
        if tlbld:
            pte = yield dut.d_out.pte
            print ("    DCACHE PTE %x -> %x" % (pte, addr))
            yield dut.d_in.done.eq(1)
            yield
            yield dut.d_in.done.eq(0)
            continue

        if addr not in mem:
            print ("    DCACHE LOOKUP FAIL %x" % (addr))
            stop = True
            return

        yield
        data = mem[addr]
        yield dut.d_in.data.eq(data)
        print ("    DCACHE GET %x data %x" % (addr, data))
        yield dut.d_in.done.eq(1)
        yield
        yield dut.d_in.done.eq(0)


def mmu_wait(dut):
    global stop
    while not stop: # wait for dc_valid / err
        d_valid = yield (dut.d_out.valid)
        if d_valid:
            tlbld = yield (dut.d_out.tlbld)
            addr = yield (dut.d_out.addr)
            print ("addr %x tlbld %d" % (addr, tlbld))
        l_done = yield (dut.l_out.done)
        l_err = yield (dut.l_out.err)
        l_badtree = yield (dut.l_out.badtree)
        l_permerr = yield (dut.l_out.perm_error)
        l_rc_err = yield (dut.l_out.rc_error)
        l_segerr = yield (dut.l_out.segerr)
        l_invalid = yield (dut.l_out.invalid)
        if (l_done or l_err or l_badtree or
            l_permerr or l_rc_err or l_segerr or l_invalid):
            break
        yield
        yield dut.l_in.valid.eq(0) # data already in MMU by now
        yield dut.l_in.mtspr.eq(0) # captured by RegStage(s)
        yield dut.l_in.load.eq(0)  # can reset everything safely


def mmu_sim(dut):
    global stop

    # microwatt PRTBL = 0x12000, other test is 0x1000000
    #prtbl = 0x100000
    #pidr = 0x0
    prtbl = 0x12000
    pidr = 0x1

    # MMU MTSPR set prtbl
    yield dut.l_in.mtspr.eq(1)
    yield dut.l_in.sprn[9].eq(1) # totally fake way to set SPR=prtbl
    yield dut.l_in.rs.eq(prtbl) # set process table
    yield dut.l_in.valid.eq(1)
    yield from mmu_wait(dut)
    yield
    yield dut.l_in.sprn.eq(0)
    yield dut.l_in.rs.eq(0)
    yield

    prtbl = yield (dut.rin.prtbl)
    print ("prtbl after MTSPR %x" % prtbl)
    assert prtbl == prtbl

    if True: # microwatt test set PIDR
        # MMU MTSPR set PIDR = 1
        yield dut.l_in.mtspr.eq(1)
        yield dut.l_in.sprn[9].eq(0) # totally fake way to set SPR=pidr
        yield dut.l_in.rs.eq(pidr) # set process table
        yield dut.l_in.valid.eq(1)
        yield from mmu_wait(dut)
        yield
        yield dut.l_in.sprn.eq(0)
        yield dut.l_in.rs.eq(0)
        yield

    #yield dut.rin.prtbl.eq(0x1000000) # manually set process table
    #yield

    #addr = 0x10000  # original test
    #addr = 0x124108  # microwatt mmu.bin test 2
    addr = 0x10b0d8  # microwatt mmu.bin test 4

    # MMU PTE request
    yield dut.l_in.load.eq(1)
    yield dut.l_in.priv.eq(1)
    yield dut.l_in.addr.eq(addr)
    yield dut.l_in.valid.eq(1)
    yield from mmu_wait(dut)

    addr = yield dut.d_out.addr
    pte = yield dut.d_out.pte
    tlb_ld = yield dut.d_out.tlbld
    l_done = yield (dut.l_out.done)
    l_err = yield (dut.l_out.err)
    l_badtree = yield (dut.l_out.badtree)
    print ("translated done %d err %d badtree %d "
           "addr %x pte %x tlb_ld %d" % \
               (l_done, l_err, l_badtree, addr, pte, tlb_ld))

    yield
    yield dut.l_in.priv.eq(0)
    yield dut.l_in.addr.eq(0)


    stop = True


def test_mmu():
    dut = MMU()
    vl = rtlil.convert(dut, ports=[])#dut.ports())
    with open("test_mmu.il", "w") as f:
        f.write(vl)

    m = Module()
    m.submodules.mmu = dut

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    sim.add_sync_process(wrap(mmu_sim(dut)))
    sim.add_sync_process(wrap(dcache_get(dut)))
    with sim.write_vcd('test_mmu.vcd'):
        sim.run()

if __name__ == '__main__':
    test_mmu()
