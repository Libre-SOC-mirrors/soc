"""DCache

based on Anton Blanchard microwatt dcache.vhdl

"""

from enum import Enum, unique

from nmigen import Module, Signal, Elaboratable, Cat, Repl, Array, Const
from nmigen.cli import main
from nmutil.iocontrol import RecordObject
from nmigen.utils import log2_int
from nmigen.cli import rtlil


from soc.experiment.mem_types import (LoadStore1ToDCacheType,
                                     DCacheToLoadStore1Type,
                                     MMUToDCacheType,
                                     DCacheToMMUType)

from soc.experiment.wb_types import (WB_ADDR_BITS, WB_DATA_BITS, WB_SEL_BITS,
                                WBAddrType, WBDataType, WBSelType,
                                WBMasterOut, WBSlaveOut,
                                WBMasterOutVector, WBSlaveOutVector,
                                WBIOMasterOut, WBIOSlaveOut)

from soc.experiment.cache_ram import CacheRam
from soc.experiment.plru import PLRU


# TODO: make these parameters of DCache at some point
LINE_SIZE = 64    # Line size in bytes
NUM_LINES = 32    # Number of lines in a set
NUM_WAYS = 4      # Number of ways
TLB_SET_SIZE = 64 # L1 DTLB entries per set
TLB_NUM_WAYS = 2  # L1 DTLB number of sets
TLB_LG_PGSZ = 12  # L1 DTLB log_2(page_size)
LOG_LENGTH = 0    # Non-zero to enable log data collection

# BRAM organisation: We never access more than
#     -- WB_DATA_BITS at a time so to save
#     -- resources we make the array only that wide, and
#     -- use consecutive indices for to make a cache "line"
#     --
#     -- ROW_SIZE is the width in bytes of the BRAM
#     -- (based on WB, so 64-bits)
ROW_SIZE = WB_DATA_BITS // 8;

# ROW_PER_LINE is the number of row (wishbone
# transactions) in a line
ROW_PER_LINE = LINE_SIZE // ROW_SIZE

# BRAM_ROWS is the number of rows in BRAM needed
# to represent the full dcache
BRAM_ROWS = NUM_LINES * ROW_PER_LINE


# Bit fields counts in the address

# REAL_ADDR_BITS is the number of real address
# bits that we store
REAL_ADDR_BITS = 56

# ROW_BITS is the number of bits to select a row
ROW_BITS = log2_int(BRAM_ROWS)

# ROW_LINE_BITS is the number of bits to select
# a row within a line
ROW_LINE_BITS = log2_int(ROW_PER_LINE)

# LINE_OFF_BITS is the number of bits for
# the offset in a cache line
LINE_OFF_BITS = log2_int(LINE_SIZE)

# ROW_OFF_BITS is the number of bits for
# the offset in a row
ROW_OFF_BITS = log2_int(ROW_SIZE)

# INDEX_BITS is the number if bits to
# select a cache line
INDEX_BITS = log2_int(NUM_LINES)

# SET_SIZE_BITS is the log base 2 of the set size
SET_SIZE_BITS = LINE_OFF_BITS + INDEX_BITS

# TAG_BITS is the number of bits of
# the tag part of the address
TAG_BITS = REAL_ADDR_BITS - SET_SIZE_BITS

# TAG_WIDTH is the width in bits of each way of the tag RAM
TAG_WIDTH = TAG_BITS + 7 - ((TAG_BITS + 7) % 8)

# WAY_BITS is the number of bits to select a way
WAY_BITS = log2_int(NUM_WAYS)

# Example of layout for 32 lines of 64 bytes:
#
# ..  tag    |index|  line  |
# ..         |   row   |    |
# ..         |     |---|    | ROW_LINE_BITS  (3)
# ..         |     |--- - --| LINE_OFF_BITS (6)
# ..         |         |- --| ROW_OFF_BITS  (3)
# ..         |----- ---|    | ROW_BITS      (8)
# ..         |-----|        | INDEX_BITS    (5)
# .. --------|              | TAG_BITS      (45)

TAG_RAM_WIDTH = TAG_WIDTH * NUM_WAYS

def CacheTagArray():
    return Array(Signal(TAG_RAM_WIDTH) for x in range(NUM_LINES))

def CacheValidBitsArray():
    return Array(Signal(INDEX_BITS) for x in range(NUM_LINES))

def RowPerLineValidArray():
    return Array(Signal() for x in range(ROW_PER_LINE))

# L1 TLB
TLB_SET_BITS     = log2_int(TLB_SET_SIZE)
TLB_WAY_BITS     = log2_int(TLB_NUM_WAYS)
TLB_EA_TAG_BITS  = 64 - (TLB_LG_PGSZ + TLB_SET_BITS)
TLB_TAG_WAY_BITS = TLB_NUM_WAYS * TLB_EA_TAG_BITS
TLB_PTE_BITS     = 64
TLB_PTE_WAY_BITS = TLB_NUM_WAYS * TLB_PTE_BITS;

assert (LINE_SIZE % ROW_SIZE) == 0, "LINE_SIZE not multiple of ROW_SIZE"
assert (LINE_SIZE % 2) == 0, "LINE_SIZE not power of 2"
assert (NUM_LINES % 2) == 0, "NUM_LINES not power of 2"
assert (ROW_PER_LINE % 2) == 0, "ROW_PER_LINE not power of 2"
assert ROW_BITS == (INDEX_BITS + ROW_LINE_BITS), "geometry bits don't add up"
assert (LINE_OFF_BITS == ROW_OFF_BITS + ROW_LINE_BITS), \
        "geometry bits don't add up"
assert REAL_ADDR_BITS == (TAG_BITS + INDEX_BITS + LINE_OFF_BITS), \
        "geometry bits don't add up"
assert REAL_ADDR_BITS == (TAG_BITS + ROW_BITS + ROW_OFF_BITS), \
         "geometry bits don't add up"
assert 64 == WB_DATA_BITS, "Can't yet handle wb width that isn't 64-bits"
assert SET_SIZE_BITS <= TLB_LG_PGSZ, "Set indexed by virtual address"


def TLBValidBitsArray():
    return Array(Signal(TLB_NUM_WAYS) for x in range(TLB_SET_SIZE))

def TLBTagsArray():
    return Array(Signal(TLB_TAG_WAY_BITS) for x in range (TLB_SET_SIZE))

def TLBPtesArray():
    return Array(Signal(TLB_PTE_WAY_BITS) for x in range(TLB_SET_SIZE))

def HitWaySet():
    return Array(Signal(NUM_WAYS) for x in range(TLB_NUM_WAYS))

# Cache RAM interface
def CacheRamOut():
    return Array(Signal(WB_DATA_BITS) for x in range(NUM_WAYS))

# PLRU output interface
def PLRUOut():
    return Array(Signal(WAY_BITS) for x in range(NUM_LINES))

# TLB PLRU output interface
def TLBPLRUOut():
    return Array(Signal(TLB_WAY_BITS) for x in range(TLB_SET_SIZE))

# Helper functions to decode incoming requests
#
# Return the cache line index (tag index) for an address
def get_index(addr):
    return addr[LINE_OFF_BITS:SET_SIZE_BITS]

# Return the cache row index (data memory) for an address
def get_row(addr):
    return addr[ROW_OFF_BITS:SET_SIZE_BITS]

# Return the index of a row within a line
def get_row_of_line(row):
    return row[:ROW_LINE_BITS]

# Returns whether this is the last row of a line
def is_last_row_addr(addr, last):
    return addr[ROW_OFF_BITS:LINE_OFF_BITS] == last

# Returns whether this is the last row of a line
def is_last_row(row, last):
    return get_row_of_line(row) == last

# Return the next row in the current cache line. We use a
# dedicated function in order to limit the size of the
# generated adder to be only the bits within a cache line
# (3 bits with default settings)
def next_row(row):
    row_v = row[0:ROW_LINE_BITS] + 1
    return Cat(row_v[:ROW_LINE_BITS], row[ROW_LINE_BITS:])

# Get the tag value from the address
def get_tag(addr):
    return addr[SET_SIZE_BITS:REAL_ADDR_BITS]

# Read a tag from a tag memory row
def read_tag(way, tagset):
    return tagset[way *TAG_WIDTH:way * TAG_WIDTH + TAG_BITS]

# Read a TLB tag from a TLB tag memory row
def read_tlb_tag(way, tags):
    j = way * TLB_EA_TAG_BITS
    return tags.bit_select(j, TLB_EA_TAG_BITS)

# Write a TLB tag to a TLB tag memory row
def write_tlb_tag(way, tags, tag):
    return read_tlb_tag(way, tags).eq(tag)

# Read a PTE from a TLB PTE memory row
def read_tlb_pte(way, ptes):
    j = way * TLB_PTE_BITS
    return ptes.bit_select(j, TLB_PTE_BITS)

def write_tlb_pte(way, ptes,newpte):
    return read_tlb_pte(way, ptes).eq(newpte)


# Record for storing permission, attribute, etc. bits from a PTE
class PermAttr(RecordObject):
    def __init__(self):
        super().__init__()
        self.reference = Signal()
        self.changed   = Signal()
        self.nocache   = Signal()
        self.priv      = Signal()
        self.rd_perm   = Signal()
        self.wr_perm   = Signal()


def extract_perm_attr(pte):
    pa = PermAttr()
    pa.reference = pte[8]
    pa.changed   = pte[7]
    pa.nocache   = pte[5]
    pa.priv      = pte[3]
    pa.rd_perm   = pte[2]
    pa.wr_perm   = pte[1]
    return pa;


# Type of operation on a "valid" input
@unique
class Op(Enum):
    OP_NONE       = 0
    OP_BAD        = 1 # NC cache hit, TLB miss, prot/RC failure
    OP_STCX_FAIL  = 2 # conditional store w/o reservation
    OP_LOAD_HIT   = 3 # Cache hit on load
    OP_LOAD_MISS  = 4 # Load missing cache
    OP_LOAD_NC    = 5 # Non-cachable load
    OP_STORE_HIT  = 6 # Store hitting cache
    OP_STORE_MISS = 7 # Store missing cache


# Cache state machine
@unique
class State(Enum):
    IDLE             = 0 # Normal load hit processing
    RELOAD_WAIT_ACK  = 1 # Cache reload wait ack
    STORE_WAIT_ACK   = 2 # Store wait ack
    NC_LOAD_WAIT_ACK = 3 # Non-cachable load wait ack


# Dcache operations:
#
# In order to make timing, we use the BRAMs with
# an output buffer, which means that the BRAM
# output is delayed by an extra cycle.
#
# Thus, the dcache has a 2-stage internal pipeline
# for cache hits with no stalls.
#
# All other operations are handled via stalling
# in the first stage.
#
# The second stage can thus complete a hit at the same
# time as the first stage emits a stall for a complex op.
#
# Stage 0 register, basically contains just the latched request

class RegStage0(RecordObject):
    def __init__(self):
        super().__init__()
        self.req     = LoadStore1ToDCacheType()
        self.tlbie   = Signal()
        self.doall   = Signal()
        self.tlbld   = Signal()
        self.mmu_req = Signal() # indicates source of request


class MemAccessRequest(RecordObject):
    def __init__(self):
        super().__init__()
        self.op        = Signal(Op)
        self.valid     = Signal()
        self.dcbz      = Signal()
        self.real_addr = Signal(REAL_ADDR_BITS)
        self.data      = Signal(64)
        self.byte_sel  = Signal(8)
        self.hit_way   = Signal(WAY_BITS)
        self.same_tag  = Signal()
        self.mmu_req   = Signal()


# First stage register, contains state for stage 1 of load hits
# and for the state machine used by all other operations
class RegStage1(RecordObject):
    def __init__(self):
        super().__init__()
        # Info about the request
        self.full             = Signal() # have uncompleted request
        self.mmu_req          = Signal() # request is from MMU
        self.req              = MemAccessRequest()

        # Cache hit state
        self.hit_way          = Signal(WAY_BITS)
        self.hit_load_valid   = Signal()
        self.hit_index        = Signal(NUM_LINES)
        self.cache_hit        = Signal()

        # TLB hit state
        self.tlb_hit          = Signal()
        self.tlb_hit_way      = Signal(TLB_NUM_WAYS)
        self.tlb_hit_index    = Signal(TLB_WAY_BITS)

        # 2-stage data buffer for data forwarded from writes to reads
        self.forward_data1    = Signal(64)
        self.forward_data2    = Signal(64)
        self.forward_sel1     = Signal(8)
        self.forward_valid1   = Signal()
        self.forward_way1     = Signal(WAY_BITS)
        self.forward_row1     = Signal(ROW_BITS)
        self.use_forward1     = Signal()
        self.forward_sel      = Signal(8)

        # Cache miss state (reload state machine)
        self.state            = Signal(State)
        self.dcbz             = Signal()
        self.write_bram       = Signal()
        self.write_tag        = Signal()
        self.slow_valid       = Signal()
        self.wb               = WBMasterOut()
        self.reload_tag       = Signal(TAG_BITS)
        self.store_way        = Signal(WAY_BITS)
        self.store_row        = Signal(ROW_BITS)
        self.store_index      = Signal(INDEX_BITS)
        self.end_row_ix       = Signal(log2_int(ROW_LINE_BITS, False))
        self.rows_valid       = RowPerLineValidArray()
        self.acks_pending     = Signal(3)
        self.inc_acks         = Signal()
        self.dec_acks         = Signal()

        # Signals to complete (possibly with error)
        self.ls_valid         = Signal()
        self.ls_error         = Signal()
        self.mmu_done         = Signal()
        self.mmu_error        = Signal()
        self.cache_paradox    = Signal()

        # Signal to complete a failed stcx.
        self.stcx_fail        = Signal()


# Reservation information
class Reservation(RecordObject):
    def __init__(self):
        super().__init__()
        self.valid = Signal()
        self.addr  = Signal(64-LINE_OFF_BITS)


class DCache(Elaboratable):
    """Set associative dcache write-through
    TODO (in no specific order):
    * See list in icache.vhdl
    * Complete load misses on the cycle when WB data comes instead of
      at the end of line (this requires dealing with requests coming in
      while not idle...)
    """
    def __init__(self):
        self.d_in      = LoadStore1ToDCacheType()
        self.d_out     = DCacheToLoadStore1Type()

        self.m_in      = MMUToDCacheType()
        self.m_out     = DCacheToMMUType()

        self.stall_out = Signal()

        self.wb_out    = WBMasterOut()
        self.wb_in     = WBSlaveOut()

        self.log_out   = Signal(20)

    def stage_0(self, m, r0, r1, r0_full):
        """Latch the request in r0.req as long as we're not stalling
        """
        comb = m.d.comb
        sync = m.d.sync
        d_in, d_out, m_in = self.d_in, self.d_out, self.m_in

        r = RegStage0()

        # TODO, this goes in unit tests and formal proofs
        with m.If(~(d_in.valid & m_in.valid)):
            #sync += Display("request collision loadstore vs MMU")
            pass

        with m.If(m_in.valid):
            sync += r.req.valid.eq(1)
            sync += r.req.load.eq(~(m_in.tlbie | m_in.tlbld))
            sync += r.req.dcbz.eq(0)
            sync += r.req.nc.eq(0)
            sync += r.req.reserve.eq(0)
            sync += r.req.virt_mode.eq(1)
            sync += r.req.priv_mode.eq(1)
            sync += r.req.addr.eq(m_in.addr)
            sync += r.req.data.eq(m_in.pte)
            sync += r.req.byte_sel.eq(~0) # Const -1 sets all to 0b111....
            sync += r.tlbie.eq(m_in.tlbie)
            sync += r.doall.eq(m_in.doall)
            sync += r.tlbld.eq(m_in.tlbld)
            sync += r.mmu_req.eq(1)
        with m.Else():
            sync += r.req.eq(d_in)
            sync += r.tlbie.eq(0)
            sync += r.doall.eq(0)
            sync += r.tlbld.eq(0)
            sync += r.mmu_req.eq(0)
            with m.If(~(r1.full & r0_full)):
                sync += r0.eq(r)
                sync += r0_full.eq(r.req.valid)

    def tlb_read(self, m, r0_stall, tlb_valid_way,
                 tlb_tag_way, tlb_pte_way, dtlb_valid_bits,
                 dtlb_tags, dtlb_ptes):
        """TLB
        Operates in the second cycle on the request latched in r0.req.
        TLB updates write the entry at the end of the second cycle.
        """
        comb = m.d.comb
        sync = m.d.sync
        m_in, d_in = self.m_in, self.d_in

        index    = Signal(TLB_SET_BITS)
        addrbits = Signal(TLB_SET_BITS)

        amin = TLB_LG_PGSZ
        amax = TLB_LG_PGSZ + TLB_SET_BITS

        with m.If(m_in.valid):
            comb += addrbits.eq(m_in.addr[amin : amax])
        with m.Else():
            comb += addrbits.eq(d_in.addr[amin : amax])
        comb += index.eq(addrbits)

        # If we have any op and the previous op isn't finished,
        # then keep the same output for next cycle.
        with m.If(~r0_stall):
            sync += tlb_valid_way.eq(dtlb_valid_bits[index])
            sync += tlb_tag_way.eq(dtlb_tags[index])
            sync += tlb_pte_way.eq(dtlb_ptes[index])

    def maybe_tlb_plrus(self, m, r1, tlb_plru_victim, acc, acc_en, lru):
        """Generate TLB PLRUs
        """
        comb = m.d.comb
        sync = m.d.sync

        with m.If(TLB_NUM_WAYS > 1):
            for i in range(TLB_SET_SIZE):
                # TLB PLRU interface
                tlb_plru        = PLRU(TLB_WAY_BITS)
                setattr(m.submodules, "maybe_plru_%d" % i, tlb_plru)
                tlb_plru_acc    = Signal(TLB_WAY_BITS)
                tlb_plru_acc_en = Signal()
                tlb_plru_out    = Signal(TLB_WAY_BITS)

                comb += tlb_plru.acc.eq(tlb_plru_acc)
                comb += tlb_plru.acc_en.eq(tlb_plru_acc_en)
                comb += tlb_plru.lru.eq(tlb_plru_out)

                # PLRU interface
                with m.If(r1.tlb_hit_index == i):
                    comb += tlb_plru.acc_en.eq(r1.tlb_hit)
                with m.Else():
                    comb += tlb_plru.acc_en.eq(0)
                comb += tlb_plru.acc.eq(r1.tlb_hit_way)

                comb += tlb_plru_victim[i].eq(tlb_plru.lru)

    def tlb_search(self, m, tlb_req_index, r0, r0_valid,
                   tlb_valid_way, tlb_tag_way, tlb_hit_way,
                   tlb_pte_way, pte, tlb_hit, valid_ra, perm_attr, ra):

        comb = m.d.comb
        sync = m.d.sync

        hitway = Signal(TLB_WAY_BITS)
        hit    = Signal()
        eatag  = Signal(TLB_EA_TAG_BITS)

        TLB_LG_END = TLB_LG_PGSZ + TLB_SET_BITS
        comb += tlb_req_index.eq(r0.req.addr[TLB_LG_PGSZ : TLB_LG_END])
        comb += eatag.eq(r0.req.addr[TLB_LG_END : 64 ])

        for i in range(TLB_NUM_WAYS):
            with m.If(tlb_valid_way[i]
                      & read_tlb_tag(i, tlb_tag_way) == eatag):
                comb += hitway.eq(i)
                comb += hit.eq(1)

        comb += tlb_hit.eq(hit & r0_valid)
        comb += tlb_hit_way.eq(hitway)

        with m.If(tlb_hit):
            comb += pte.eq(read_tlb_pte(hitway, tlb_pte_way))
        with m.Else():
            comb += pte.eq(0)
        comb += valid_ra.eq(tlb_hit | ~r0.req.virt_mode)
        with m.If(r0.req.virt_mode):
            comb += ra.eq(Cat(Const(0, ROW_OFF_BITS),
                              r0.req.addr[ROW_OFF_BITS:TLB_LG_PGSZ],
                              pte[TLB_LG_PGSZ:REAL_ADDR_BITS]))
            comb += perm_attr.eq(extract_perm_attr(pte))
        with m.Else():
            comb += ra.eq(Cat(Const(0, ROW_OFF_BITS),
                              r0.req.addr[ROW_OFF_BITS:REAL_ADDR_BITS]))

            comb += perm_attr.reference.eq(1)
            comb += perm_attr.changed.eq(1)
            comb += perm_attr.priv.eq(1)
            comb += perm_attr.nocache.eq(0)
            comb += perm_attr.rd_perm.eq(1)
            comb += perm_attr.wr_perm.eq(1)

    def tlb_update(self, m, r0_valid, r0, dtlb_valid_bits, tlb_req_index,
                    tlb_hit_way, tlb_hit, tlb_plru_victim, tlb_tag_way,
                    dtlb_tags, tlb_pte_way, dtlb_ptes):

        comb = m.d.comb
        sync = m.d.sync

        tlbie    = Signal()
        tlbwe    = Signal()
        repl_way = Signal(TLB_WAY_BITS)
        eatag    = Signal(TLB_EA_TAG_BITS)
        tagset   = Signal(TLB_TAG_WAY_BITS)
        pteset   = Signal(TLB_PTE_WAY_BITS)

        vb = Signal(TLB_NUM_WAYS)

        comb += tlbie.eq(r0_valid & r0.tlbie)
        comb += tlbwe.eq(r0_valid & r0.tlbld)
        sync += vb.eq(dtlb_valid_bits[tlb_req_index])

        with m.If(tlbie & r0.doall):
            # clear all valid bits at once
            for i in range(TLB_SET_SIZE):
                sync += dtlb_valid_bits[i].eq(0)

        with m.Elif(tlbie):
            with m.If(tlb_hit):
                sync += vb.bit_select(tlb_hit_way, 1).eq(Const(0, 1))
        with m.Elif(tlbwe):
            with m.If(tlb_hit):
                comb += repl_way.eq(tlb_hit_way)
            with m.Else():
                comb += repl_way.eq(tlb_plru_victim[tlb_req_index])
            comb += eatag.eq(r0.req.addr[TLB_LG_PGSZ + TLB_SET_BITS:64])
            sync += tagset.eq(tlb_tag_way)
            sync += write_tlb_tag(repl_way, tagset, eatag)
            sync += dtlb_tags[tlb_req_index].eq(tagset)
            sync += pteset.eq(tlb_pte_way)
            sync += write_tlb_pte(repl_way, pteset, r0.req.data)
            sync += dtlb_ptes[tlb_req_index].eq(pteset)
            sync += vb.bit_select(repl_way, 1).eq(1)

    def maybe_plrus(self, m, r1, plru_victim):
        """Generate PLRUs
        """
        comb = m.d.comb
        sync = m.d.sync

        for i in range(NUM_LINES):
            # PLRU interface
            plru        = PLRU(TLB_WAY_BITS)
            setattr(m.submodules, "plru%d" % i, plru)
            plru_acc    = Signal(WAY_BITS)
            plru_acc_en = Signal()
            plru_out    = Signal(WAY_BITS)

            comb += plru.acc.eq(plru_acc)
            comb += plru.acc_en.eq(plru_acc_en)
            comb += plru_out.eq(plru.lru_o)

            with m.If(r1.hit_index == i):
                comb += plru_acc_en.eq(r1.cache_hit)

            comb += plru_acc.eq(r1.hit_way)
            comb += plru_victim[i].eq(plru_out)

    def cache_tag_read(self, m, r0_stall, req_index, cache_tag_set, cache_tags):
        """Cache tag RAM read port
        """
        comb = m.d.comb
        sync = m.d.sync
        m_in, d_in = self.m_in, self.d_in

        index = Signal(INDEX_BITS)

        with m.If(r0_stall):
            comb += index.eq(req_index)
        with m.Elif(m_in.valid):
            comb += index.eq(get_index(m_in.addr))
        with m.Else():
            comb += index.eq(get_index(d_in.addr))
        sync += cache_tag_set.eq(cache_tags[index])

    def dcache_request(self, m, r0, ra, req_index, req_row, req_tag,
                       r0_valid, r1, cache_valid_bits, replace_way,
                       use_forward1_next, use_forward2_next,
                       req_hit_way, plru_victim, rc_ok, perm_attr,
                       valid_ra, perm_ok, access_ok, req_op, req_go,
                       tlb_pte_way,
                       tlb_hit, tlb_hit_way, tlb_valid_way, cache_tag_set,
                       cancel_store, req_same_tag, r0_stall, early_req_row):
        """Cache request parsing and hit detection
        """

        comb = m.d.comb
        sync = m.d.sync
        m_in, d_in = self.m_in, self.d_in

        is_hit      = Signal()
        hit_way     = Signal(WAY_BITS)
        op          = Signal(Op)
        opsel       = Signal(3)
        go          = Signal()
        nc          = Signal()
        s_hit       = Signal()
        s_tag       = Signal(TAG_BITS)
        s_pte       = Signal(TLB_PTE_BITS)
        s_ra        = Signal(REAL_ADDR_BITS)
        hit_set     = Signal(TLB_NUM_WAYS)
        hit_way_set = HitWaySet()
        rel_matches = Signal(TLB_NUM_WAYS)
        rel_match   = Signal()

        # Extract line, row and tag from request
        comb += req_index.eq(get_index(r0.req.addr))
        comb += req_row.eq(get_row(r0.req.addr))
        comb += req_tag.eq(get_tag(ra))

        comb += go.eq(r0_valid & ~(r0.tlbie | r0.tlbld) & ~r1.ls_error)

        # Test if pending request is a hit on any way
        # In order to make timing in virtual mode,
        # when we are using the TLB, we compare each
        # way with each of the real addresses from each way of
        # the TLB, and then decide later which match to use.

        with m.If(r0.req.virt_mode):
            comb += rel_matches.eq(0)
            for j in range(TLB_NUM_WAYS):
                comb += s_pte.eq(read_tlb_pte(j, tlb_pte_way))
                comb += s_ra.eq(Cat(r0.req.addr[0:TLB_LG_PGSZ],
                                    s_pte[TLB_LG_PGSZ:REAL_ADDR_BITS]))
                comb += s_tag.eq(get_tag(s_ra))

                for i in range(NUM_WAYS):
                    with m.If(go & cache_valid_bits[req_index][i] &
                              read_tag(i, cache_tag_set) == s_tag
                              & tlb_valid_way[j]):
                        comb += hit_way_set[j].eq(i)
                        comb += s_hit.eq(1)
                comb += hit_set[j].eq(s_hit)
                with m.If(s_tag == r1.reload_tag):
                    comb += rel_matches[j].eq(1)
            with m.If(tlb_hit):
                comb += is_hit.eq(hit_set.bit_select(tlb_hit_way, 1))
                comb += hit_way.eq(hit_way_set[tlb_hit_way])
                comb += rel_match.eq(rel_matches.bit_select(tlb_hit_way, 1))
        with m.Else():
            comb += s_tag.eq(get_tag(r0.req.addr))
            for i in range(NUM_WAYS):
                with m.If(go & cache_valid_bits[req_index][i] &
                          read_tag(i, cache_tag_set) == s_tag):
                    comb += hit_way.eq(i)
                    comb += is_hit.eq(1)
            with m.If(s_tag == r1.reload_tag):
                comb += rel_match.eq(1)
        comb += req_same_tag.eq(rel_match)

        # See if the request matches the line currently being reloaded
        with m.If((r1.state == State.RELOAD_WAIT_ACK) &
                  (req_index == r1.store_index) & rel_match):
            # For a store, consider this a hit even if the row isn't
            # valid since it will be by the time we perform the store.
            # For a load, check the appropriate row valid bit.
            valid = r1.rows_valid[req_row % ROW_PER_LINE]
            comb += is_hit.eq(~r0.req.load | valid)
            comb += hit_way.eq(replace_way)

        # Whether to use forwarded data for a load or not
        comb += use_forward1_next.eq(0)
        with m.If((get_row(r1.req.real_addr) == req_row) &
                  (r1.req.hit_way == hit_way)):
            # Only need to consider r1.write_bram here, since if we
            # are writing refill data here, then we don't have a
            # cache hit this cycle on the line being refilled.
            # (There is the possibility that the load following the
            # load miss that started the refill could be to the old
            # contents of the victim line, since it is a couple of
            # cycles after the refill starts before we see the updated
            # cache tag. In that case we don't use the bypass.)
            comb += use_forward1_next.eq(r1.write_bram)
        comb += use_forward2_next.eq(0)
        with m.If((r1.forward_row1 == req_row) & (r1.forward_way1 == hit_way)):
            comb += use_forward2_next.eq(r1.forward_valid1)

        # The way that matched on a hit
        comb += req_hit_way.eq(hit_way)

        # The way to replace on a miss
        with m.If(r1.write_tag):
            replace_way.eq(plru_victim[r1.store_index])
        with m.Else():
            comb += replace_way.eq(r1.store_way)

        # work out whether we have permission for this access
        # NB we don't yet implement AMR, thus no KUAP
        comb += rc_ok.eq(perm_attr.reference
                         & (r0.req.load | perm_attr.changed)
                )
        comb += perm_ok.eq((r0.req.priv_mode | ~perm_attr.priv)
                           & perm_attr.wr_perm
                           | (r0.req.load & perm_attr.rd_perm)
                          )
        comb += access_ok.eq(valid_ra & perm_ok & rc_ok)
        # Combine the request and cache hit status to decide what
        # operation needs to be done
        comb += nc.eq(r0.req.nc | perm_attr.nocache)
        comb += op.eq(Op.OP_NONE)
        with m.If(go):
            with m.If(~access_ok):
                comb += op.eq(Op.OP_BAD)
            with m.Elif(cancel_store):
                comb += op.eq(Op.OP_STCX_FAIL)
            with m.Else():
                comb += opsel.eq(Cat(is_hit, nc, r0.req.load))
                with m.Switch(opsel):
                    with m.Case(0b101):
                        comb += op.eq(Op.OP_LOAD_HIT)
                    with m.Case(0b100):
                        comb += op.eq(Op.OP_LOAD_MISS)
                    with m.Case(0b110):
                        comb += op.eq(Op.OP_LOAD_NC)
                    with m.Case(0b001):
                        comb += op.eq(Op.OP_STORE_HIT)
                    with m.Case(0b000):
                        comb += op.eq(Op.OP_STORE_MISS)
                    with m.Case(0b010):
                        comb += op.eq(Op.OP_STORE_MISS)
                    with m.Case(0b011):
                        comb += op.eq(Op.OP_BAD)
                    with m.Case(0b111):
                        comb += op.eq(Op.OP_BAD)
                    with m.Default():
                        comb += op.eq(Op.OP_NONE)
        comb += req_op.eq(op)
        comb += req_go.eq(go)

        # Version of the row number that is valid one cycle earlier
        # in the cases where we need to read the cache data BRAM.
        # If we're stalling then we need to keep reading the last
        # row requested.
        with m.If(~r0_stall):
            with m.If(m_in.valid):
                comb += early_req_row.eq(get_row(m_in.addr))
            with m.Else():
                comb += early_req_row.eq(get_row(d_in.addr))
        with m.Else():
            comb += early_req_row.eq(req_row)

    def reservation_comb(self, m, cancel_store, set_rsrv, clear_rsrv,
                         r0_valid, r0, reservation):
        """Handle load-with-reservation and store-conditional instructions
        """
        comb = m.d.comb
        sync = m.d.sync

        with m.If(r0_valid & r0.req.reserve):

            # XXX generate alignment interrupt if address
            # is not aligned XXX or if r0.req.nc = '1'
            with m.If(r0.req.load):
                comb += set_rsrv.eq(1) # load with reservation
            with m.Else():
                comb += clear_rsrv.eq(1) # store conditional
                with m.If(~reservation.valid | r0.req.addr[LINE_OFF_BITS:64]):
                    comb += cancel_store.eq(1)

    def reservation_reg(self, m, r0_valid, access_ok, set_rsrv, clear_rsrv,
                        reservation, r0):

        comb = m.d.comb
        sync = m.d.sync

        with m.If(r0_valid & access_ok):
            with m.If(clear_rsrv):
                sync += reservation.valid.eq(0)
            with m.Elif(set_rsrv):
                sync += reservation.valid.eq(1)
                sync += reservation.addr.eq(r0.req.addr[LINE_OFF_BITS:64])

    def writeback_control(self, m, r1, cache_out):
        """Return data for loads & completion control logic
        """
        comb = m.d.comb
        sync = m.d.sync
        d_out, m_out = self.d_out, self.m_out

        data_out = Signal(64)
        data_fwd = Signal(64)

        # Use the bypass if are reading the row that was
        # written 1 or 2 cycles ago, including for the
        # slow_valid = 1 case (i.e. completing a load
        # miss or a non-cacheable load).
        with m.If(r1.use_forward1):
            comb += data_fwd.eq(r1.forward_data1)
        with m.Else():
            comb += data_fwd.eq(r1.forward_data2)

        comb += data_out.eq(cache_out[r1.hit_way])

        for i in range(8):
            with m.If(r1.forward_sel[i]):
                dsel = data_fwd.word_select(i, 8)
                comb += data_out.word_select(i, 8).eq(dsel)

        comb += d_out.valid.eq(r1.ls_valid)
        comb += d_out.data.eq(data_out)
        comb += d_out.store_done.eq(~r1.stcx_fail)
        comb += d_out.error.eq(r1.ls_error)
        comb += d_out.cache_paradox.eq(r1.cache_paradox)

        # Outputs to MMU
        comb += m_out.done.eq(r1.mmu_done)
        comb += m_out.err.eq(r1.mmu_error)
        comb += m_out.data.eq(data_out)

        # We have a valid load or store hit or we just completed
        # a slow op such as a load miss, a NC load or a store
        #
        # Note: the load hit is delayed by one cycle. However it
        # can still not collide with r.slow_valid (well unless I
        # miscalculated) because slow_valid can only be set on a
        # subsequent request and not on its first cycle (the state
        # machine must have advanced), which makes slow_valid
        # at least 2 cycles from the previous hit_load_valid.

        # Sanity: Only one of these must be set in any given cycle

        if False: # TODO: need Display to get this to work
            assert (r1.slow_valid & r1.stcx_fail) != 1, \
            "unexpected slow_valid collision with stcx_fail"

            assert ((r1.slow_valid | r1.stcx_fail) | r1.hit_load_valid) != 1, \
             "unexpected hit_load_delayed collision with slow_valid"

        with m.If(~r1.mmu_req):
            # Request came from loadstore1...
            # Load hit case is the standard path
            with m.If(r1.hit_load_valid):
                #Display(f"completing load hit data={data_out}")
                pass

            # error cases complete without stalling
            with m.If(r1.ls_error):
                # Display("completing ld/st with error")
                pass

            # Slow ops (load miss, NC, stores)
            with m.If(r1.slow_valid):
                #Display(f"completing store or load miss data={data_out}")
                pass

        with m.Else():
            # Request came from MMU
            with m.If(r1.hit_load_valid):
                # Display(f"completing load hit to MMU, data={m_out.data}")
                pass
            # error cases complete without stalling
            with m.If(r1.mmu_error):
                #Display("combpleting MMU ld with error")
                pass

            # Slow ops (i.e. load miss)
            with m.If(r1.slow_valid):
                #Display("completing MMU load miss, data={m_out.data}")
                pass

    def rams(self, m, r1, early_req_row, cache_out, replace_way):
        """rams
        Generate a cache RAM for each way. This handles the normal
        reads, writes from reloads and the special store-hit update
        path as well.

        Note: the BRAMs have an extra read buffer, meaning the output
        is pipelined an extra cycle. This differs from the
        icache. The writeback logic needs to take that into
        account by using 1-cycle delayed signals for load hits.
        """
        comb = m.d.comb
        wb_in = self.wb_in

        for i in range(NUM_WAYS):
            do_read  = Signal()
            rd_addr  = Signal(ROW_BITS)
            do_write = Signal()
            wr_addr  = Signal(ROW_BITS)
            wr_data  = Signal(WB_DATA_BITS)
            wr_sel   = Signal(ROW_SIZE)
            wr_sel_m = Signal(ROW_SIZE)
            _d_out   = Signal(WB_DATA_BITS)

            way = CacheRam(ROW_BITS, WB_DATA_BITS, True)
            setattr(m.submodules, "cacheram_%d" % i, way)

            comb += way.rd_en.eq(do_read)
            comb += way.rd_addr.eq(rd_addr)
            comb += _d_out.eq(way.rd_data_o)
            comb += way.wr_sel.eq(wr_sel_m)
            comb += way.wr_addr.eq(wr_addr)
            comb += way.wr_data.eq(wr_data)

            # Cache hit reads
            comb += do_read.eq(1)
            comb += rd_addr.eq(early_req_row)
            comb += cache_out[i].eq(_d_out)

            # Write mux:
            #
            # Defaults to wishbone read responses (cache refill)
            #
            # For timing, the mux on wr_data/sel/addr is not
            # dependent on anything other than the current state.

            with m.If(r1.write_bram):
                # Write store data to BRAM.  This happens one
                # cycle after the store is in r0.
                comb += wr_data.eq(r1.req.data)
                comb += wr_sel.eq(r1.req.byte_sel)
                comb += wr_addr.eq(get_row(r1.req.real_addr))

                with m.If(i == r1.req.hit_way):
                    comb += do_write.eq(1)
            with m.Else():
                # Otherwise, we might be doing a reload or a DCBZ
                with m.If(r1.dcbz):
                    comb += wr_data.eq(0)
                with m.Else():
                    comb += wr_data.eq(wb_in.dat)
                comb += wr_addr.eq(r1.store_row)
                comb += wr_sel.eq(~0) # all 1s

            with m.If((r1.state == State.RELOAD_WAIT_ACK)
                      & wb_in.ack & (replace_way == i)):
                comb += do_write.eq(1)

                # Mask write selects with do_write since BRAM
                # doesn't have a global write-enable
                with m.If(do_write):
                    comb += wr_sel_m.eq(wr_sel)

    # Cache hit synchronous machine for the easy case.
    # This handles load hits.
    # It also handles error cases (TLB miss, cache paradox)
    def dcache_fast_hit(self, m, req_op, r0_valid, r0, r1,
                        req_hit_way, req_index, access_ok,
                        tlb_hit, tlb_hit_way, tlb_req_index):

        comb = m.d.comb
        sync = m.d.sync

        with m.If(req_op != Op.OP_NONE):
            #Display(f"op:{req_op} addr:{r0.req.addr} nc: {r0.req.nc}" \
            #      f"idx:{req_index} tag:{req_tag} way: {req_hit_way}"
            #     )
            pass

        with m.If(r0_valid):
            sync += r1.mmu_req.eq(r0.mmu_req)

        # Fast path for load/store hits.
        # Set signals for the writeback controls.
        sync += r1.hit_way.eq(req_hit_way)
        sync += r1.hit_index.eq(req_index)

        with m.If(req_op == Op.OP_LOAD_HIT):
            sync += r1.hit_load_valid.eq(1)
        with m.Else():
            sync += r1.hit_load_valid.eq(0)

        with m.If((req_op == Op.OP_LOAD_HIT) | (req_op == Op.OP_STORE_HIT)):
            sync += r1.cache_hit.eq(1)
        with m.Else():
            sync += r1.cache_hit.eq(0)

        with m.If(req_op == Op.OP_BAD):
            # Display(f"Signalling ld/st error valid_ra={valid_ra}"
            #      f"rc_ok={rc_ok} perm_ok={perm_ok}"
            sync += r1.ls_error.eq(~r0.mmu_req)
            sync += r1.mmu_error.eq(r0.mmu_req)
            sync += r1.cache_paradox.eq(access_ok)

            with m.Else():
                sync += r1.ls_error.eq(0)
                sync += r1.mmu_error.eq(0)
                sync += r1.cache_paradox.eq(0)

        with m.If(req_op == Op.OP_STCX_FAIL):
            r1.stcx_fail.eq(1)
        with m.Else():
            sync += r1.stcx_fail.eq(0)

        # Record TLB hit information for updating TLB PLRU
        sync += r1.tlb_hit.eq(tlb_hit)
        sync += r1.tlb_hit_way.eq(tlb_hit_way)
        sync += r1.tlb_hit_index.eq(tlb_req_index)

    # Memory accesses are handled by this state machine:
    #
    #   * Cache load miss/reload (in conjunction with "rams")
    #   * Load hits for non-cachable forms
    #   * Stores (the collision case is handled in "rams")
    #
    # All wishbone requests generation is done here.
    # This machine operates at stage 1.
    def dcache_slow(self, m, r1, use_forward1_next, use_forward2_next,
                    cache_valid_bits, r0, replace_way,
                    req_hit_way, req_same_tag,
                    r0_valid, req_op, cache_tag, req_go, ra):

        comb = m.d.comb
        sync = m.d.sync
        wb_in = self.wb_in

        req         = MemAccessRequest()
        acks        = Signal(3)
        adjust_acks = Signal(3)
        stbs_done = Signal()

        sync += r1.use_forward1.eq(use_forward1_next)
        sync += r1.forward_sel.eq(0)

        with m.If(use_forward1_next):
            sync += r1.forward_sel.eq(r1.req.byte_sel)
        with m.Elif(use_forward2_next):
            sync += r1.forward_sel.eq(r1.forward_sel1)

        sync += r1.forward_data2.eq(r1.forward_data1)
        with m.If(r1.write_bram):
            sync += r1.forward_data1.eq(r1.req.data)
            sync += r1.forward_sel1.eq(r1.req.byte_sel)
            sync += r1.forward_way1.eq(r1.req.hit_way)
            sync += r1.forward_row1.eq(get_row(r1.req.real_addr))
            sync += r1.forward_valid1.eq(1)
        with m.Else():
            with m.If(r1.dcbz):
                sync += r1.forward_data1.eq(0)
            with m.Else():
                sync += r1.forward_data1.eq(wb_in.dat)
            sync += r1.forward_sel1.eq(~0) # all 1s
            sync += r1.forward_way1.eq(replace_way)
            sync += r1.forward_row1.eq(r1.store_row)
            sync += r1.forward_valid1.eq(0)

        # One cycle pulses reset
        sync += r1.slow_valid.eq(0)
        sync += r1.write_bram.eq(0)
        sync += r1.inc_acks.eq(0)
        sync += r1.dec_acks.eq(0)

        sync += r1.ls_valid.eq(0)
        # complete tlbies and TLB loads in the third cycle
        sync += r1.mmu_done.eq(r0_valid & (r0.tlbie | r0.tlbld))

        with m.If((req_op == Op.OP_LOAD_HIT)
                  | (req_op == Op.OP_STCX_FAIL)):
            with m.If(~r0.mmu_req):
                sync += r1.ls_valid.eq(1)
            with m.Else():
                sync += r1.mmu_done.eq(1)

        with m.If(r1.write_tag):
            # Store new tag in selected way
            for i in range(NUM_WAYS):
                with m.If(i == replace_way):
                    ct = cache_tag[r1.store_index].word_select(i, TAG_WIDTH)
                    sync += ct.eq(r1.reload_tag)
            sync += r1.store_way.eq(replace_way)
            sync += r1.write_tag.eq(0)

        # Take request from r1.req if there is one there,
        # else from req_op, ra, etc.
        with m.If(r1.full):
            comb += req.eq(r1.req)
        with m.Else():
            comb += req.op.eq(req_op)
            comb += req.valid.eq(req_go)
            comb += req.mmu_req.eq(r0.mmu_req)
            comb += req.dcbz.eq(r0.req.dcbz)
            comb += req.real_addr.eq(ra)

            with m.If(~r0.req.dcbz):
                comb += req.data.eq(r0.req.data)
            with m.Else():
                comb += req.data.eq(0)

            # Select all bytes for dcbz
            # and for cacheable loads
            with m.If(r0.req.dcbz | (r0.req.load & ~r0.req.nc)):
                comb += req.byte_sel.eq(~0) # all 1s
            with m.Else():
                comb += req.byte_sel.eq(r0.req.byte_sel)
            comb += req.hit_way.eq(req_hit_way)
            comb += req.same_tag.eq(req_same_tag)

            # Store the incoming request from r0,
            # if it is a slow request
            # Note that r1.full = 1 implies req_op = OP_NONE
            with m.If((req_op == Op.OP_LOAD_MISS)
                      | (req_op == Op.OP_LOAD_NC)
                      | (req_op == Op.OP_STORE_MISS)
                      | (req_op == Op.OP_STORE_HIT)):
                sync += r1.req.eq(req)
                sync += r1.full.eq(1)

        # Main state machine
        with m.Switch(r1.state):

            with m.Case(State.IDLE):
# XXX check 'left downto.  probably means len(r1.wb.adr)
#                     r1.wb.adr <= req.real_addr(
#                                   r1.wb.adr'left downto 0
#                                  );
                sync += r1.wb.adr.eq(req.real_addr)
                sync += r1.wb.sel.eq(req.byte_sel)
                sync += r1.wb.dat.eq(req.data)
                sync += r1.dcbz.eq(req.dcbz)

                # Keep track of our index and way
                # for subsequent stores.
                sync += r1.store_index.eq(get_index(req.real_addr))
                sync += r1.store_row.eq(get_row(req.real_addr))
                sync += r1.end_row_ix.eq(
                         get_row_of_line(get_row(req.real_addr))
                        )
                sync += r1.reload_tag.eq(get_tag(req.real_addr))
                sync += r1.req.same_tag.eq(1)

                with m.If(req.op == Op.OP_STORE_HIT):
                    sync += r1.store_way.eq(req.hit_way)

                # Reset per-row valid bits,
                # ready for handling OP_LOAD_MISS
                for i in range(ROW_PER_LINE):
                    sync += r1.rows_valid[i].eq(0)

                with m.Switch(req.op):
                    with m.Case(Op.OP_LOAD_HIT):
                        # stay in IDLE state
                        pass

                    with m.Case(Op.OP_LOAD_MISS):
                        #Display(f"cache miss real addr:" \
                        #      f"{req_real_addr}" \
                        #      f" idx:{get_index(req_real_addr)}" \
                        #      f" tag:{get_tag(req.real_addr)}")
                        pass

                        # Start the wishbone cycle
                        sync += r1.wb.we.eq(0)
                        sync += r1.wb.cyc.eq(1)
                        sync += r1.wb.stb.eq(1)

                        # Track that we had one request sent
                        sync += r1.state.eq(State.RELOAD_WAIT_ACK)
                        sync += r1.write_tag.eq(1)

                    with m.Case(Op.OP_LOAD_NC):
                        sync += r1.wb.cyc.eq(1)
                        sync += r1.wb.stb.eq(1)
                        sync += r1.wb.we.eq(0)
                        sync += r1.state.eq(State.NC_LOAD_WAIT_ACK)

                    with m.Case(Op.OP_STORE_HIT, Op.OP_STORE_MISS):
                        with m.If(~req.dcbz):
                            sync += r1.state.eq(State.STORE_WAIT_ACK)
                            sync += r1.acks_pending.eq(1)
                            sync += r1.full.eq(0)
                            sync += r1.slow_valid.eq(1)

                            with m.If(~req.mmu_req):
                                sync += r1.ls_valid.eq(1)
                            with m.Else():
                                sync += r1.mmu_done.eq(1)

                            with m.If(req.op == Op.OP_STORE_HIT):
                                sync += r1.write_bram.eq(1)
                        with m.Else():
                            sync += r1.state.eq(State.RELOAD_WAIT_ACK)

                            with m.If(req.op == Op.OP_STORE_MISS):
                                sync += r1.write_tag.eq(1)

                        sync += r1.wb.we.eq(1)
                        sync += r1.wb.cyc.eq(1)
                        sync += r1.wb.stb.eq(1)

                    # OP_NONE and OP_BAD do nothing
                    # OP_BAD & OP_STCX_FAIL were
                    # handled above already
                    with m.Case(Op.OP_NONE):
                        pass
                    with m.Case(Op.OP_BAD):
                        pass
                    with m.Case(Op.OP_STCX_FAIL):
                        pass

            with m.Case(State.RELOAD_WAIT_ACK):
                # Requests are all sent if stb is 0
                comb += stbs_done.eq(~r1.wb.stb)

                with m.If(~wb_in.stall & ~stbs_done):
                    # That was the last word?
                    # We are done sending.
                    # Clear stb and set stbs_done
                    # so we can handle an eventual
                    # last ack on the same cycle.
                    with m.If(is_last_row_addr(
                              r1.wb.adr, r1.end_row_ix)):
                        sync += r1.wb.stb.eq(0)
                        comb += stbs_done.eq(0)

                    # Calculate the next row address in the current cache line
                    rarange = r1.wb.adr[ROW_OFF_BITS : LINE_OFF_BITS]
                    sync += rarange.eq(rarange + 1)

                # Incoming acks processing
                sync += r1.forward_valid1.eq(wb_in.ack)
                with m.If(wb_in.ack):
                    # XXX needs an Array bit-accessor here
                    sync += r1.rows_valid[r1.store_row % ROW_PER_LINE].eq(1)

                    # If this is the data we were looking for,
                    # we can complete the request next cycle.
                    # Compare the whole address in case the
                    # request in r1.req is not the one that
                    # started this refill.
                    with m.If(r1.full & r1.req.same_tag &
                              ((r1.dcbz & r1.req.dcbz) |
                               (~r1.dcbz & (r1.req.op == Op.OP_LOAD_MISS))) &
                                (r1.store_row == get_row(r1.req.real_addr))):
                        sync += r1.full.eq(0)
                        sync += r1.slow_valid.eq(1)
                        with m.If(~r1.mmu_req):
                            sync += r1.ls_valid.eq(1)
                        with m.Else():
                            sync += r1.mmu_done.eq(1)
                        sync += r1.forward_sel.eq(~0) # all 1s
                        sync += r1.use_forward1.eq(1)

                    # Check for completion
                    with m.If(stbs_done & is_last_row(r1.store_row,
                                                      r1.end_row_ix)):
                        # Complete wishbone cycle
                        sync += r1.wb.cyc.eq(0)

                        # Cache line is now valid
                        cv = Signal(INDEX_BITS)
                        sync += cv.eq(cache_valid_bits[r1.store_index])
                        sync += cv.bit_select(r1.store_way, 1).eq(1)
                        sync += r1.state.eq(State.IDLE)

                    # Increment store row counter
                    sync += r1.store_row.eq(next_row(r1.store_row))

            with m.Case(State.STORE_WAIT_ACK):
                comb += stbs_done.eq(~r1.wb.stb)
                comb += acks.eq(r1.acks_pending)

                with m.If(r1.inc_acks != r1.dec_acks):
                    with m.If(r1.inc_acks):
                        comb += adjust_acks.eq(acks + 1)
                    with m.Else():
                        comb += adjust_acks.eq(acks - 1)
                with m.Else():
                    comb += adjust_acks.eq(acks)

                sync += r1.acks_pending.eq(adjust_acks)

                # Clear stb when slave accepted request
                with m.If(~wb_in.stall):
                    # See if there is another store waiting
                    # to be done which is in the same real page.
                    with m.If(req.valid):
                        ra = req.real_addr[0:SET_SIZE_BITS]
                        sync += r1.wb.adr[0:SET_SIZE_BITS].eq(ra)
                        sync += r1.wb.dat.eq(req.data)
                        sync += r1.wb.sel.eq(req.byte_sel)

                    with m.Elif((adjust_acks < 7) & req.same_tag &
                                ((req.op == Op.OP_STORE_MISS)
                                 | (req.op == Op.OP_STORE_HIT))):
                        sync += r1.wb.stb.eq(1)
                        comb += stbs_done.eq(0)

                        with m.If(req.op == Op.OP_STORE_HIT):
                            sync += r1.write_bram.eq(1)
                        sync += r1.full.eq(0)
                        sync += r1.slow_valid.eq(1)

                        # Store requests never come from the MMU
                        sync += r1.ls_valid.eq(1)
                        comb += stbs_done.eq(0)
                        sync += r1.inc_acks.eq(1)
                    with m.Else():
                        sync += r1.wb.stb.eq(0)
                        comb += stbs_done.eq(1)

                # Got ack ? See if complete.
                with m.If(wb_in.ack):
                    with m.If(stbs_done & (adjust_acks == 1)):
                        sync += r1.state.eq(State.IDLE)
                        sync += r1.wb.cyc.eq(0)
                        sync += r1.wb.stb.eq(0)
                    sync += r1.dec_acks.eq(1)

            with m.Case(State.NC_LOAD_WAIT_ACK):
                # Clear stb when slave accepted request
                with m.If(~wb_in.stall):
                    sync += r1.wb.stb.eq(0)

                # Got ack ? complete.
                with m.If(wb_in.ack):
                    sync += r1.state.eq(State.IDLE)
                    sync += r1.full.eq(0)
                    sync += r1.slow_valid.eq(1)

                    with m.If(~r1.mmu_req):
                        sync += r1.ls_valid.eq(1)
                    with m.Else():
                        sync += r1.mmu_done.eq(1)

                    sync += r1.forward_sel.eq(~0) # all 1s
                    sync += r1.use_forward1.eq(1)
                    sync += r1.wb.cyc.eq(0)
                    sync += r1.wb.stb.eq(0)

    def dcache_log(self, m, r1, valid_ra, tlb_hit_way, stall_out):

        sync = m.d.sync
        d_out, wb_in, log_out = self.d_out, self.wb_in, self.log_out

        sync += log_out.eq(Cat(r1.state[:3], valid_ra, tlb_hit_way[:3],
                               stall_out, req_op[:3], d_out.valid, d_out.error,
                               r1.wb.cyc, r1.wb.stb, wb_in.ack, wb_in.stall,
                               r1.wb.adr[3:6]))

    def elaborate(self, platform):

        m = Module()
        comb = m.d.comb

        # Storage. Hopefully "cache_rows" is a BRAM, the rest is LUTs
        cache_tags       = CacheTagArray()
        cache_tag_set    = Signal(TAG_RAM_WIDTH)
        cache_valid_bits = CacheValidBitsArray()

        # TODO attribute ram_style : string;
        # TODO attribute ram_style of cache_tags : signal is "distributed";

        """note: these are passed to nmigen.hdl.Memory as "attributes".
           don't know how, just that they are.
        """
        dtlb_valid_bits = TLBValidBitsArray()
        dtlb_tags       = TLBTagsArray()
        dtlb_ptes       = TLBPtesArray()
        # TODO attribute ram_style of
        #  dtlb_tags : signal is "distributed";
        # TODO attribute ram_style of
        #  dtlb_ptes : signal is "distributed";

        r0      = RegStage0()
        r0_full = Signal()

        r1 = RegStage1()

        reservation = Reservation()

        # Async signals on incoming request
        req_index    = Signal(INDEX_BITS)
        req_row      = Signal(ROW_BITS)
        req_hit_way  = Signal(WAY_BITS)
        req_tag      = Signal(TAG_BITS)
        req_op       = Signal(Op)
        req_data     = Signal(64)
        req_same_tag = Signal()
        req_go       = Signal()

        early_req_row     = Signal(ROW_BITS)

        cancel_store      = Signal()
        set_rsrv          = Signal()
        clear_rsrv        = Signal()

        r0_valid          = Signal()
        r0_stall          = Signal()

        use_forward1_next = Signal()
        use_forward2_next = Signal()

        cache_out         = CacheRamOut()

        plru_victim       = PLRUOut()
        replace_way       = Signal(WAY_BITS)

        # Wishbone read/write/cache write formatting signals
        bus_sel           = Signal(8)

        # TLB signals
        tlb_tag_way   = Signal(TLB_TAG_WAY_BITS)
        tlb_pte_way   = Signal(TLB_PTE_WAY_BITS)
        tlb_valid_way = Signal(TLB_NUM_WAYS)
        tlb_req_index = Signal(TLB_SET_BITS)
        tlb_hit       = Signal()
        tlb_hit_way   = Signal(TLB_WAY_BITS)
        pte           = Signal(TLB_PTE_BITS)
        ra            = Signal(REAL_ADDR_BITS)
        valid_ra      = Signal()
        perm_attr     = PermAttr()
        rc_ok         = Signal()
        perm_ok       = Signal()
        access_ok     = Signal()

        tlb_plru_victim = TLBPLRUOut()

        # we don't yet handle collisions between loadstore1 requests
        # and MMU requests
        comb += self.m_out.stall.eq(0)

        # Hold off the request in r0 when r1 has an uncompleted request
        comb += r0_stall.eq(r0_full & r1.full)
        comb += r0_valid.eq(r0_full & ~r1.full)
        comb += self.stall_out.eq(r0_stall)

        # Wire up wishbone request latch out of stage 1
        comb += self.wb_out.eq(r1.wb)

        # call sub-functions putting everything together, using shared
        # signals established above
        self.stage_0(m, r0, r1, r0_full)
        self.tlb_read(m, r0_stall, tlb_valid_way,
                      tlb_tag_way, tlb_pte_way, dtlb_valid_bits,
                      dtlb_tags, dtlb_ptes)
        self.tlb_search(m, tlb_req_index, r0, r0_valid,
                        tlb_valid_way, tlb_tag_way, tlb_hit_way,
                        tlb_pte_way, pte, tlb_hit, valid_ra, perm_attr, ra)
        self.tlb_update(m, r0_valid, r0, dtlb_valid_bits, tlb_req_index,
                        tlb_hit_way, tlb_hit, tlb_plru_victim, tlb_tag_way,
                        dtlb_tags, tlb_pte_way, dtlb_ptes)
        self.maybe_plrus(m, r1, plru_victim)
        self.cache_tag_read(m, r0_stall, req_index, cache_tag_set, cache_tags)
        self.dcache_request(m, r0, ra, req_index, req_row, req_tag,
                           r0_valid, r1, cache_valid_bits, replace_way,
                           use_forward1_next, use_forward2_next,
                           req_hit_way, plru_victim, rc_ok, perm_attr,
                           valid_ra, perm_ok, access_ok, req_op, req_go,
                           tlb_pte_way,
                           tlb_hit, tlb_hit_way, tlb_valid_way, cache_tag_set,
                           cancel_store, req_same_tag, r0_stall, early_req_row)
        self.reservation_comb(m, cancel_store, set_rsrv, clear_rsrv,
                           r0_valid, r0, reservation)
        self.reservation_reg(m, r0_valid, access_ok, set_rsrv, clear_rsrv,
                           reservation, r0)
        self.writeback_control(m, r1, cache_out)
        self.rams(m, r1, early_req_row, cache_out, replace_way)
        self.dcache_fast_hit(m, req_op, r0_valid, r0, r1,
                        req_hit_way, req_index, access_ok,
                        tlb_hit, tlb_hit_way, tlb_req_index)
        self.dcache_slow(m, r1, use_forward1_next, use_forward2_next,
                    cache_valid_bits, r0, replace_way,
                    req_hit_way, req_same_tag,
                         r0_valid, req_op, cache_tags, req_go, ra)
        #self.dcache_log(m, r1, valid_ra, tlb_hit_way, stall_out)

        return m


# dcache_tb.vhdl
#
# entity dcache_tb is
# end dcache_tb;
#
# architecture behave of dcache_tb is
#     signal clk          : std_ulogic;
#     signal rst          : std_ulogic;
#
#     signal d_in         : Loadstore1ToDcacheType;
#     signal d_out        : DcacheToLoadstore1Type;
#
#     signal m_in         : MmuToDcacheType;
#     signal m_out        : DcacheToMmuType;
#
#     signal wb_bram_in   : wishbone_master_out;
#     signal wb_bram_out  : wishbone_slave_out;
#
#     constant clk_period : time := 10 ns;
# begin
#     dcache0: entity work.dcache
#         generic map(
#
#             LINE_SIZE => 64,
#             NUM_LINES => 4
#             )
#         port map(
#             clk => clk,
#             rst => rst,
#             d_in => d_in,
#             d_out => d_out,
#             m_in => m_in,
#             m_out => m_out,
#             wishbone_out => wb_bram_in,
#             wishbone_in => wb_bram_out
#             );
#
#     -- BRAM Memory slave
#     bram0: entity work.wishbone_bram_wrapper
#         generic map(
#             MEMORY_SIZE   => 1024,
#             RAM_INIT_FILE => "icache_test.bin"
#             )
#         port map(
#             clk => clk,
#             rst => rst,
#             wishbone_in => wb_bram_in,
#             wishbone_out => wb_bram_out
#             );
#
#     clk_process: process
#     begin
#         clk <= '0';
#         wait for clk_period/2;
#         clk <= '1';
#         wait for clk_period/2;
#     end process;
#
#     rst_process: process
#     begin
#         rst <= '1';
#         wait for 2*clk_period;
#         rst <= '0';
#         wait;
#     end process;
#
#     stim: process
#     begin
#     -- Clear stuff
#     d_in.valid <= '0';
#     d_in.load <= '0';
#     d_in.nc <= '0';
#     d_in.addr <= (others => '0');
#     d_in.data <= (others => '0');
#         m_in.valid <= '0';
#         m_in.addr <= (others => '0');
#         m_in.pte <= (others => '0');
#
#         wait for 4*clk_period;
#     wait until rising_edge(clk);
#
#     -- Cacheable read of address 4
#     d_in.load <= '1';
#     d_in.nc <= '0';
#         d_in.addr <= x"0000000000000004";
#         d_in.valid <= '1';
#     wait until rising_edge(clk);
#         d_in.valid <= '0';
#
#     wait until rising_edge(clk) and d_out.valid = '1';
#         assert d_out.data = x"0000000100000000"
#         report "data @" & to_hstring(d_in.addr) &
#         "=" & to_hstring(d_out.data) &
#         " expected 0000000100000000"
#         severity failure;
# --      wait for clk_period;
#
#     -- Cacheable read of address 30
#     d_in.load <= '1';
#     d_in.nc <= '0';
#         d_in.addr <= x"0000000000000030";
#         d_in.valid <= '1';
#     wait until rising_edge(clk);
#         d_in.valid <= '0';
#
#     wait until rising_edge(clk) and d_out.valid = '1';
#         assert d_out.data = x"0000000D0000000C"
#         report "data @" & to_hstring(d_in.addr) &
#         "=" & to_hstring(d_out.data) &
#         " expected 0000000D0000000C"
#         severity failure;
#
#     -- Non-cacheable read of address 100
#     d_in.load <= '1';
#     d_in.nc <= '1';
#         d_in.addr <= x"0000000000000100";
#         d_in.valid <= '1';
#     wait until rising_edge(clk);
#     d_in.valid <= '0';
#     wait until rising_edge(clk) and d_out.valid = '1';
#         assert d_out.data = x"0000004100000040"
#         report "data @" & to_hstring(d_in.addr) &
#         "=" & to_hstring(d_out.data) &
#         " expected 0000004100000040"
#         severity failure;
#
#     wait until rising_edge(clk);
#     wait until rising_edge(clk);
#     wait until rising_edge(clk);
#     wait until rising_edge(clk);
#
#     std.env.finish;
#     end process;
# end;
def dcache_sim(dut):
    # clear stuff
    yield dut.d_in.valid.eq(0)
    yield dut.d_in.load.eq(0)
    yield dut.d_in.nc.eq(0)
    yield dut.d_in.adrr.eq(0)
    yield dut.d_in.data.eq(0)
    yield dut.m_in.valid.eq(0)
    yield dut.m_in.addr.eq(0)
    yield dut.m_in.pte.eq(0)
    # wait 4 * clk_period
    yield
    yield
    yield
    yield
    # wait_until rising_edge(clk)
    yield
    # Cacheable read of address 4
    yield dut.d_in.load.eq(1)
    yield dut.d_in.nc.eq(0)
    yield dut.d_in.addr.eq(Const(0x0000000000000004, 64))
    yield dut.d_in.valid.eq(1)
    # wait-until rising_edge(clk)
    yield
    yield dut.d_in.valid.eq(0)
    yield
    while not (yield dut.d_out.valid):
        yield
    assert dut.d_out.data == 0x0000000100000000, \
        f"data @ {dut.d_in.addr}={dut.d_in.data} expected 0000000100000000"


    # Cacheable read of address 30
    yield dut.d_in.load.eq(1)
    yield dut.d_in.nc.eq(0)
    yield dut.d_in.addr.eq(Const(0x0000000000000030, 64))
    yield dut.d_in.valid.eq(1)
    yield
    yield dut.d_in.valid.eq(0)
    yield
    while not (yield dut.d_out.valid):
        yield
    assert dut.d_out.data == 0x0000000D0000000C, \
        f"data @{dut.d_in.addr}={dut.d_out.data} expected 0000000D0000000C"

    # Non-cacheable read of address 100
    yield dut.d_in.load.eq(1)
    yield dut.d_in.nc.eq(1)
    yield dut.d_in.addr.eq(Const(0x0000000000000100, 64))
    yield dut.d_in.valid.eq(1)
    yield
    yield dut.d_in.valid.eq(0)
    yield
    while not (yield dut.d_out.valid):
        yield
    assert dut.d_out.data == 0x0000004100000040, \
        f"data @ {dut.d_in.addr}={dut.d_out.data} expected 0000004100000040"

    yield
    yield
    yield
    yield


def test_dcache():
    dut = DCache()
    vl = rtlil.convert(dut, ports=[])
    with open("test_dcache.il", "w") as f:
        f.write(vl)

    #run_simulation(dut, dcache_sim(), vcd_name='test_dcache.vcd')

if __name__ == '__main__':
    test_dcache()

