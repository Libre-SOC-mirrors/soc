"""DCache

based on Anton Blanchard microwatt dcache.vhdl

note that the microwatt dcache wishbone interface expects "stall".
for simplicity at the moment this is hard-coded to cyc & ~ack.
see WB4 spec, p84, section 5.2.1

IMPORTANT: for store, the data is sampled the cycle AFTER the "valid"
is raised.  sigh

Links:

* https://libre-soc.org/3d_gpu/architecture/set_associative_cache.jpg
* https://bugs.libre-soc.org/show_bug.cgi?id=469
* https://libre-soc.org/irclog-microwatt/%23microwatt.2021-12-07.log.html
  (discussion about brams for ECP5)

"""

import sys

from nmutil.gtkw import write_gtkw

sys.setrecursionlimit(1000000)

from enum import Enum, unique

from nmigen import (Module, Signal, Elaboratable, Cat, Repl, Array, Const,
                    Record, Memory)
from nmutil.util import Display
from nmigen.lib.coding import Decoder

from copy import deepcopy
from random import randint, seed

from nmigen_soc.wishbone.bus import Interface

from nmigen.cli import main
from nmutil.iocontrol import RecordObject
from nmigen.utils import log2_int
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
from soc.experiment.plru import PLRU, PLRUs
#from nmutil.plru import PLRU, PLRUs

# for test
from soc.bus.sram import SRAM
from nmigen import Memory
from nmigen.cli import rtlil

# NOTE: to use cxxsim, export NMIGEN_SIM_MODE=cxxsim from the shell
# Also, check out the cxxsim nmigen branch, and latest yosys from git
from nmutil.sim_tmp_alternative import Simulator

from nmutil.util import wrap


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
#     -- use consecutive indices to make a cache "line"
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

print ("ROW_SIZE", ROW_SIZE)
print ("ROW_PER_LINE", ROW_PER_LINE)
print ("BRAM_ROWS", BRAM_ROWS)
print ("NUM_WAYS", NUM_WAYS)

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
layout = f"""\
  DCache Layout:
 |.. -----------------------| REAL_ADDR_BITS ({REAL_ADDR_BITS})
  ..         |--------------| SET_SIZE_BITS ({SET_SIZE_BITS})
  ..  tag    |index|  line  |
  ..         |   row   |    |
  ..         |     |---|    | ROW_LINE_BITS ({ROW_LINE_BITS})
  ..         |     |--- - --| LINE_OFF_BITS ({LINE_OFF_BITS})
  ..         |         |- --| ROW_OFF_BITS  ({ROW_OFF_BITS})
  ..         |----- ---|    | ROW_BITS      ({ROW_BITS})
  ..         |-----|        | INDEX_BITS    ({INDEX_BITS})
  .. --------|              | TAG_BITS      ({TAG_BITS})
"""
print (layout)
print ("Dcache TAG %d IDX %d ROW_BITS %d ROFF %d LOFF %d RLB %d" % \
            (TAG_BITS, INDEX_BITS, ROW_BITS,
             ROW_OFF_BITS, LINE_OFF_BITS, ROW_LINE_BITS))
print ("index @: %d-%d" % (LINE_OFF_BITS, SET_SIZE_BITS))
print ("row @: %d-%d" % (LINE_OFF_BITS, ROW_OFF_BITS))
print ("tag @: %d-%d width %d" % (SET_SIZE_BITS, REAL_ADDR_BITS, TAG_WIDTH))

TAG_RAM_WIDTH = TAG_WIDTH * NUM_WAYS

print ("TAG_RAM_WIDTH", TAG_RAM_WIDTH)
print ("    TAG_WIDTH", TAG_WIDTH)
print ("     NUM_WAYS", NUM_WAYS)
print ("    NUM_LINES", NUM_LINES)


def CacheTag(name=None):
    tag_layout = [('valid', NUM_WAYS),
                  ('tag', TAG_RAM_WIDTH),
                 ]
    return Record(tag_layout, name=name)


def CacheTagArray():
    return Array(CacheTag(name="tag%d" % x) for x in range(NUM_LINES))


def RowPerLineValidArray():
    return Array(Signal(name="rows_valid%d" % x) \
                        for x in range(ROW_PER_LINE))


# L1 TLB
TLB_SET_BITS     = log2_int(TLB_SET_SIZE)
TLB_WAY_BITS     = log2_int(TLB_NUM_WAYS)
TLB_EA_TAG_BITS  = 64 - (TLB_LG_PGSZ + TLB_SET_BITS)
TLB_TAG_WAY_BITS = TLB_NUM_WAYS * TLB_EA_TAG_BITS
TLB_PTE_BITS     = 64
TLB_PTE_WAY_BITS = TLB_NUM_WAYS * TLB_PTE_BITS;

def ispow2(x):
    return (1<<log2_int(x, False)) == x

assert (LINE_SIZE % ROW_SIZE) == 0, "LINE_SIZE not multiple of ROW_SIZE"
assert ispow2(LINE_SIZE), "LINE_SIZE not power of 2"
assert ispow2(NUM_LINES), "NUM_LINES not power of 2"
assert ispow2(ROW_PER_LINE), "ROW_PER_LINE not power of 2"
assert ROW_BITS == (INDEX_BITS + ROW_LINE_BITS), "geometry bits don't add up"
assert (LINE_OFF_BITS == ROW_OFF_BITS + ROW_LINE_BITS), \
        "geometry bits don't add up"
assert REAL_ADDR_BITS == (TAG_BITS + INDEX_BITS + LINE_OFF_BITS), \
        "geometry bits don't add up"
assert REAL_ADDR_BITS == (TAG_BITS + ROW_BITS + ROW_OFF_BITS), \
         "geometry bits don't add up"
assert 64 == WB_DATA_BITS, "Can't yet handle wb width that isn't 64-bits"
assert SET_SIZE_BITS <= TLB_LG_PGSZ, "Set indexed by virtual address"


def TLBHit(name):
    return Record([('valid', 1),
                   ('way', TLB_WAY_BITS)], name=name)

def TLBTagEAArray():
    return Array(Signal(TLB_EA_TAG_BITS, name="tlbtagea%d" % x) \
                for x in range (TLB_NUM_WAYS))

def TLBRecord(name):
    tlb_layout = [('valid', TLB_NUM_WAYS),
                  ('tag', TLB_TAG_WAY_BITS),
                  ('pte', TLB_PTE_WAY_BITS)
                 ]
    return Record(tlb_layout, name=name)

def TLBValidArray():
    return Array(Signal(TLB_NUM_WAYS, name="tlb_valid%d" % x)
                        for x in range(TLB_SET_SIZE))

def HitWaySet():
    return Array(Signal(WAY_BITS, name="hitway_%d" % x) \
                        for x in range(TLB_NUM_WAYS))

# Cache RAM interface
def CacheRamOut():
    return Array(Signal(WB_DATA_BITS, name="cache_out%d" % x) \
                 for x in range(NUM_WAYS))

# PLRU output interface
def PLRUOut():
    return Array(Signal(WAY_BITS, name="plru_out%d" % x) \
                for x in range(NUM_LINES))

# TLB PLRU output interface
def TLBPLRUOut():
    return Array(Signal(TLB_WAY_BITS, name="tlbplru_out%d" % x) \
                for x in range(TLB_SET_SIZE))

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
    return row[:ROW_BITS][:ROW_LINE_BITS]

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
    return tagset.word_select(way, TAG_WIDTH)[:TAG_BITS]

# Read a TLB tag from a TLB tag memory row
def read_tlb_tag(way, tags):
    return tags.word_select(way, TLB_EA_TAG_BITS)

# Write a TLB tag to a TLB tag memory row
def write_tlb_tag(way, tags, tag):
    return read_tlb_tag(way, tags).eq(tag)

# Read a PTE from a TLB PTE memory row
def read_tlb_pte(way, ptes):
    return ptes.word_select(way, TLB_PTE_BITS)

def write_tlb_pte(way, ptes, newpte):
    return read_tlb_pte(way, ptes).eq(newpte)


# Record for storing permission, attribute, etc. bits from a PTE
class PermAttr(RecordObject):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.reference = Signal()
        self.changed   = Signal()
        self.nocache   = Signal()
        self.priv      = Signal()
        self.rd_perm   = Signal()
        self.wr_perm   = Signal()


def extract_perm_attr(pte):
    pa = PermAttr()
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
    def __init__(self, name=None):
        super().__init__(name=name)
        self.req     = LoadStore1ToDCacheType(name="lsmem")
        self.tlbie   = Signal() # indicates a tlbie request (from MMU)
        self.doall   = Signal() # with tlbie, indicates flush whole TLB
        self.tlbld   = Signal() # indicates a TLB load request (from MMU)
        self.mmu_req = Signal() # indicates source of request
        self.d_valid = Signal() # indicates req.data is valid now


class MemAccessRequest(RecordObject):
    def __init__(self, name=None):
        super().__init__(name=name)
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
    def __init__(self, name=None):
        super().__init__(name=name)
        # Info about the request
        self.full             = Signal() # have uncompleted request
        self.mmu_req          = Signal() # request is from MMU
        self.req              = MemAccessRequest(name="reqmem")

        # Cache hit state
        self.hit_way          = Signal(WAY_BITS)
        self.hit_load_valid   = Signal()
        self.hit_index        = Signal(INDEX_BITS)
        self.cache_hit        = Signal()

        # TLB hit state
        self.tlb_hit          = TLBHit("tlb_hit")
        self.tlb_hit_index    = Signal(TLB_SET_BITS)

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
        self.wb               = WBMasterOut("wb")
        self.reload_tag       = Signal(TAG_BITS)
        self.store_way        = Signal(WAY_BITS)
        self.store_row        = Signal(ROW_BITS)
        self.store_index      = Signal(INDEX_BITS)
        self.end_row_ix       = Signal(ROW_LINE_BITS)
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


class DTLBUpdate(Elaboratable):
    def __init__(self):
        self.tlbie    = Signal()
        self.tlbwe    = Signal()
        self.doall    = Signal()
        self.tlb_hit     = TLBHit("tlb_hit")
        self.tlb_req_index = Signal(TLB_SET_BITS)

        self.repl_way        = Signal(TLB_WAY_BITS)
        self.eatag           = Signal(TLB_EA_TAG_BITS)
        self.pte_data        = Signal(TLB_PTE_BITS)

        # read from dtlb array
        self.tlb_read       = Signal()
        self.tlb_read_index = Signal(TLB_SET_BITS)
        self.tlb_way        = TLBRecord("o_tlb_way")

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        sync = m.d.sync

        # there are 3 parts to this:
        # QTY TLB_NUM_WAYs TAGs - of width (say) 46 bits of Effective Address
        # QTY TLB_NUM_WAYs PTEs - of width (say) 64 bits
        # "Valid" bits, one per "way", of QTY TLB_NUM_WAYs.  these cannot
        # be a Memory because they can all be cleared (tlbie, doall), i mean,
        # we _could_, in theory, by overriding the Reset Signal of the Memory,
        # hmmm....

        dtlb_valid = TLBValidArray()
        tlb_req_index = self.tlb_req_index

        print ("TLB_TAG_WAY_BITS", TLB_TAG_WAY_BITS)
        print ("     TLB_EA_TAG_BITS", TLB_EA_TAG_BITS)
        print ("        TLB_NUM_WAYS", TLB_NUM_WAYS)
        print ("TLB_PTE_WAY_BITS", TLB_PTE_WAY_BITS)
        print ("    TLB_PTE_BITS", TLB_PTE_BITS)
        print ("    TLB_NUM_WAYS", TLB_NUM_WAYS)

        # TAG and PTE Memory SRAMs. transparent, write-enables are TLB_NUM_WAYS
        tagway = Memory(depth=TLB_SET_SIZE, width=TLB_TAG_WAY_BITS)
        m.submodules.rd_tagway = rd_tagway = tagway.read_port()
        m.submodules.wr_tagway = wr_tagway = tagway.write_port(
                                    granularity=TLB_EA_TAG_BITS)

        pteway = Memory(depth=TLB_SET_SIZE, width=TLB_PTE_WAY_BITS)
        m.submodules.rd_pteway = rd_pteway = pteway.read_port()
        m.submodules.wr_pteway = wr_pteway = pteway.write_port(
                                    granularity=TLB_PTE_BITS)

        # commented out for now, can be put in if Memory.reset can be
        # used for tlbie&doall to reset the entire Memory to zero in 1 cycle
        #validm = Memory(depth=TLB_SET_SIZE, width=TLB_NUM_WAYS)
        #m.submodules.rd_valid = rd_valid = validm.read_port()
        #m.submodules.wr_valid = wr_valid = validm.write_port(
                                    #granularity=1)

        # connect up read and write addresses to Valid/PTE/TAG SRAMs
        m.d.comb += rd_pteway.addr.eq(self.tlb_read_index)
        m.d.comb += rd_tagway.addr.eq(self.tlb_read_index)
        #m.d.comb += rd_valid.addr.eq(self.tlb_read_index)
        m.d.comb += wr_tagway.addr.eq(tlb_req_index)
        m.d.comb += wr_pteway.addr.eq(tlb_req_index)
        #m.d.comb += wr_valid.addr.eq(tlb_req_index)

        updated  = Signal()
        v_updated  = Signal()
        tb_out = Signal(TLB_TAG_WAY_BITS) # tlb_way_tags_t
        db_out = Signal(TLB_NUM_WAYS)     # tlb_way_valids_t
        pb_out = Signal(TLB_PTE_WAY_BITS) # tlb_way_ptes_t
        dv = Signal(TLB_NUM_WAYS) # tlb_way_valids_t

        comb += dv.eq(dtlb_valid[tlb_req_index])
        comb += db_out.eq(dv)

        with m.If(self.tlbie & self.doall):
            # clear all valid bits at once
            # XXX hmmm, validm _could_ use Memory reset here...
            for i in range(TLB_SET_SIZE):
                sync += dtlb_valid[i].eq(0)
        with m.Elif(self.tlbie):
            # invalidate just the hit_way
            with m.If(self.tlb_hit.valid):
                comb += db_out.bit_select(self.tlb_hit.way, 1).eq(0)
                comb += v_updated.eq(1)
        with m.Elif(self.tlbwe):
            # write to the requested tag and PTE
            comb += write_tlb_tag(self.repl_way, tb_out, self.eatag)
            comb += write_tlb_pte(self.repl_way, pb_out, self.pte_data)
            # set valid bit
            comb += db_out.bit_select(self.repl_way, 1).eq(1)

            comb += updated.eq(1)
            comb += v_updated.eq(1)

        # above, sometimes valid is requested to be updated but data not
        # therefore split them out, here.  note the granularity thing matches
        # with the shift-up of the eatag/pte_data into the correct TLB way.
        # thus is it not necessary to write the entire lot, just the portion
        # being altered: hence writing the *old* copy of the row is not needed
        with m.If(updated): # PTE and TAG to be written
            comb += wr_pteway.data.eq(pb_out)
            comb += wr_pteway.en.eq(1<<self.repl_way)
            comb += wr_tagway.data.eq(tb_out)
            comb += wr_tagway.en.eq(1<<self.repl_way)
        with m.If(v_updated): # Valid to be written
            sync += dtlb_valid[tlb_req_index].eq(db_out)
            #comb += wr_valid.data.eq(db_out)
            #comb += wr_valid.en.eq(1<<self.repl_way)

        # select one TLB way, use a register here
        r_tlb_way        = TLBRecord("r_tlb_way")
        r_delay = Signal()
        sync += r_delay.eq(self.tlb_read)
        with m.If(self.tlb_read):
            sync += self.tlb_way.valid.eq(dtlb_valid[self.tlb_read_index])
        with m.If(r_delay):
            # on one clock delay, output the contents of the read port(s)
            # comb += self.tlb_way.valid.eq(rd_valid.data)
            comb += self.tlb_way.tag.eq(rd_tagway.data)
            comb += self.tlb_way.pte.eq(rd_pteway.data)
            # and also capture the (delayed) output...
            #sync += r_tlb_way.valid.eq(rd_valid.data)
            sync += r_tlb_way.tag.eq(rd_tagway.data)
            sync += r_tlb_way.pte.eq(rd_pteway.data)
        with m.Else():
            # ... so that the register can output it when no read is requested
            # it's rather overkill but better to be safe than sorry
            comb += self.tlb_way.tag.eq(r_tlb_way.tag)
            comb += self.tlb_way.pte.eq(r_tlb_way.pte)
            #comb += self.tlb_way.eq(r_tlb_way)

        return m


class DCachePendingHit(Elaboratable):

    def __init__(self, tlb_way,
                      cache_i_validdx, cache_tag_set,
                    req_addr):

        self.go          = Signal()
        self.virt_mode   = Signal()
        self.is_hit      = Signal()
        self.tlb_hit      = TLBHit("tlb_hit")
        self.hit_way     = Signal(WAY_BITS)
        self.rel_match   = Signal()
        self.req_index   = Signal(INDEX_BITS)
        self.reload_tag  = Signal(TAG_BITS)

        self.tlb_way = tlb_way
        self.cache_i_validdx = cache_i_validdx
        self.cache_tag_set = cache_tag_set
        self.req_addr = req_addr

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        sync = m.d.sync

        go = self.go
        virt_mode = self.virt_mode
        is_hit = self.is_hit
        tlb_way = self.tlb_way
        cache_i_validdx = self.cache_i_validdx
        cache_tag_set = self.cache_tag_set
        req_addr = self.req_addr
        tlb_hit = self.tlb_hit
        hit_way = self.hit_way
        rel_match = self.rel_match
        req_index = self.req_index
        reload_tag = self.reload_tag

        hit_set     = Array(Signal(name="hit_set_%d" % i) \
                                  for i in range(TLB_NUM_WAYS))
        rel_matches = Array(Signal(name="rel_matches_%d" % i) \
                                    for i in range(TLB_NUM_WAYS))
        hit_way_set = HitWaySet()

        # Test if pending request is a hit on any way
        # In order to make timing in virtual mode,
        # when we are using the TLB, we compare each
        # way with each of the real addresses from each way of
        # the TLB, and then decide later which match to use.

        with m.If(virt_mode):
            for j in range(TLB_NUM_WAYS): # tlb_num_way_t
                s_tag       = Signal(TAG_BITS, name="s_tag%d" % j)
                s_hit       = Signal(name="s_hit%d" % j)
                s_pte       = Signal(TLB_PTE_BITS, name="s_pte%d" % j)
                s_ra        = Signal(REAL_ADDR_BITS, name="s_ra%d" % j)
                # read the PTE, calc the Real Address, get tge tag
                comb += s_pte.eq(read_tlb_pte(j, tlb_way.pte))
                comb += s_ra.eq(Cat(req_addr[0:TLB_LG_PGSZ],
                                    s_pte[TLB_LG_PGSZ:REAL_ADDR_BITS]))
                comb += s_tag.eq(get_tag(s_ra))
                # for each way check tge tag against the cache tag set
                for i in range(NUM_WAYS): # way_t
                    is_tag_hit = Signal(name="is_tag_hit_%d_%d" % (j, i))
                    comb += is_tag_hit.eq(go & cache_i_validdx[i] &
                                  (read_tag(i, cache_tag_set) == s_tag)
                                  & (tlb_way.valid[j]))
                    with m.If(is_tag_hit):
                        comb += hit_way_set[j].eq(i)
                        comb += s_hit.eq(1)
                comb += hit_set[j].eq(s_hit)
                comb += rel_matches[j].eq(s_tag == reload_tag)
            with m.If(tlb_hit.valid):
                comb += is_hit.eq(hit_set[tlb_hit.way])
                comb += hit_way.eq(hit_way_set[tlb_hit.way])
                comb += rel_match.eq(rel_matches[tlb_hit.way])
        with m.Else():
            s_tag       = Signal(TAG_BITS)
            comb += s_tag.eq(get_tag(req_addr))
            for i in range(NUM_WAYS): # way_t
                is_tag_hit = Signal(name="is_tag_hit_%d" % i)
                comb += is_tag_hit.eq(go & cache_i_validdx[i] &
                          (read_tag(i, cache_tag_set) == s_tag))
                with m.If(is_tag_hit):
                    comb += hit_way.eq(i)
                    comb += is_hit.eq(1)
            with m.If(s_tag == reload_tag):
                comb += rel_match.eq(1)

        return m


class DCache(Elaboratable):
    """Set associative dcache write-through

    TODO (in no specific order):
    * See list in icache.vhdl
    * Complete load misses on the cycle when WB data comes instead of
      at the end of line (this requires dealing with requests coming in
      while not idle...)
    """
    def __init__(self, pspec=None):
        self.d_in      = LoadStore1ToDCacheType("d_in")
        self.d_out     = DCacheToLoadStore1Type("d_out")

        self.m_in      = MMUToDCacheType("m_in")
        self.m_out     = DCacheToMMUType("m_out")

        self.stall_out = Signal()

        # standard naming (wired to non-standard for compatibility)
        self.bus = Interface(addr_width=32,
                            data_width=64,
                            granularity=8,
                            features={'stall'},
                            alignment=0,
                            name="dcache")

        self.log_out   = Signal(20)

        # test if microwatt compatibility is to be enabled
        self.microwatt_compat = (hasattr(pspec, "microwatt_compat") and
                                 (pspec.microwatt_compat == True))

    def stage_0(self, m, r0, r1, r0_full):
        """Latch the request in r0.req as long as we're not stalling
        """
        comb = m.d.comb
        sync = m.d.sync
        d_in, d_out, m_in = self.d_in, self.d_out, self.m_in

        r = RegStage0("stage0")

        # TODO, this goes in unit tests and formal proofs
        with m.If(d_in.valid & m_in.valid):
            sync += Display("request collision loadstore vs MMU")

        with m.If(m_in.valid):
            comb += r.req.valid.eq(1)
            comb += r.req.load.eq(~(m_in.tlbie | m_in.tlbld))# no invalidate
            comb += r.req.dcbz.eq(0)
            comb += r.req.nc.eq(0)
            comb += r.req.reserve.eq(0)
            comb += r.req.virt_mode.eq(0)
            comb += r.req.priv_mode.eq(1)
            comb += r.req.addr.eq(m_in.addr)
            comb += r.req.data.eq(m_in.pte)
            comb += r.req.byte_sel.eq(~0) # Const -1 sets all to 0b111....
            comb += r.tlbie.eq(m_in.tlbie)
            comb += r.doall.eq(m_in.doall)
            comb += r.tlbld.eq(m_in.tlbld)
            comb += r.mmu_req.eq(1)
            comb += r.d_valid.eq(1)
            m.d.sync += Display("    DCACHE req mmu addr %x pte %x ld %d",
                                 m_in.addr, m_in.pte, r.req.load)

        with m.Else():
            comb += r.req.eq(d_in)
            comb += r.req.data.eq(0)
            comb += r.tlbie.eq(0)
            comb += r.doall.eq(0)
            comb += r.tlbld.eq(0)
            comb += r.mmu_req.eq(0)
            comb += r.d_valid.eq(0)

        with m.If((~r1.full & ~d_in.hold) | ~r0_full):
            sync += r0.eq(r)
            sync += r0_full.eq(r.req.valid)
        with m.Elif(~r0.d_valid):
            # Sample data the cycle after a request comes in from loadstore1.
            # If another request has come in already then the data will get
            # put directly into req.data below.
            sync += r0.req.data.eq(d_in.data)
            sync += r0.d_valid.eq(1)
        with m.If(d_in.valid):
            m.d.sync += Display("    DCACHE req cache "
                                "virt %d addr %x data %x ld %d",
                                 r.req.virt_mode, r.req.addr,
                                 r.req.data, r.req.load)

    def tlb_read(self, m, r0_stall, tlb_way):
        """TLB
        Operates in the second cycle on the request latched in r0.req.
        TLB updates write the entry at the end of the second cycle.
        """
        comb = m.d.comb
        sync = m.d.sync
        m_in, d_in = self.m_in, self.d_in

        addrbits = Signal(TLB_SET_BITS)

        amin = TLB_LG_PGSZ
        amax = TLB_LG_PGSZ + TLB_SET_BITS

        with m.If(m_in.valid):
            comb += addrbits.eq(m_in.addr[amin : amax])
        with m.Else():
            comb += addrbits.eq(d_in.addr[amin : amax])

        # If we have any op and the previous op isn't finished,
        # then keep the same output for next cycle.
        d = self.dtlb_update
        comb += d.tlb_read_index.eq(addrbits)
        comb += d.tlb_read.eq(~r0_stall)
        comb += tlb_way.eq(d.tlb_way)

    def maybe_tlb_plrus(self, m, r1, tlb_plru_victim, tlb_req_index):
        """Generate TLB PLRUs
        """
        comb = m.d.comb
        sync = m.d.sync

        if TLB_NUM_WAYS == 0:
            return

        # suite of PLRUs with a selection and output mechanism
        tlb_plrus = PLRUs(TLB_SET_SIZE, TLB_WAY_BITS)
        m.submodules.tlb_plrus = tlb_plrus
        comb += tlb_plrus.way.eq(r1.tlb_hit.way)
        comb += tlb_plrus.valid.eq(r1.tlb_hit.valid)
        comb += tlb_plrus.index.eq(r1.tlb_hit_index)
        comb += tlb_plrus.isel.eq(tlb_req_index) # select victim
        comb += tlb_plru_victim.eq(tlb_plrus.o_index) # selected victim

    def tlb_search(self, m, tlb_req_index, r0, r0_valid,
                   tlb_way,
                   pte, tlb_hit, valid_ra, perm_attr, ra):

        comb = m.d.comb

        hitway = Signal(TLB_WAY_BITS)
        hit    = Signal()
        eatag  = Signal(TLB_EA_TAG_BITS)

        TLB_LG_END = TLB_LG_PGSZ + TLB_SET_BITS
        comb += tlb_req_index.eq(r0.req.addr[TLB_LG_PGSZ : TLB_LG_END])
        comb += eatag.eq(r0.req.addr[TLB_LG_END : 64 ])

        for i in range(TLB_NUM_WAYS):
            is_tag_hit = Signal(name="is_tag_hit%d" % i)
            tlb_tag = Signal(TLB_EA_TAG_BITS, name="tlb_tag%d" % i)
            comb += tlb_tag.eq(read_tlb_tag(i, tlb_way.tag))
            comb += is_tag_hit.eq((tlb_way.valid[i]) & (tlb_tag == eatag))
            with m.If(is_tag_hit):
                comb += hitway.eq(i)
                comb += hit.eq(1)

        comb += tlb_hit.valid.eq(hit & r0_valid)
        comb += tlb_hit.way.eq(hitway)

        with m.If(tlb_hit.valid):
            comb += pte.eq(read_tlb_pte(hitway, tlb_way.pte))
        comb += valid_ra.eq(tlb_hit.valid | ~r0.req.virt_mode)

        with m.If(r0.req.virt_mode):
            comb += ra.eq(Cat(Const(0, ROW_OFF_BITS),
                              r0.req.addr[ROW_OFF_BITS:TLB_LG_PGSZ],
                              pte[TLB_LG_PGSZ:REAL_ADDR_BITS]))
            comb += perm_attr.reference.eq(pte[8])
            comb += perm_attr.changed.eq(pte[7])
            comb += perm_attr.nocache.eq(pte[5])
            comb += perm_attr.priv.eq(pte[3])
            comb += perm_attr.rd_perm.eq(pte[2])
            comb += perm_attr.wr_perm.eq(pte[1])
        with m.Else():
            comb += ra.eq(Cat(Const(0, ROW_OFF_BITS),
                              r0.req.addr[ROW_OFF_BITS:REAL_ADDR_BITS]))
            comb += perm_attr.reference.eq(1)
            comb += perm_attr.changed.eq(1)
            comb += perm_attr.nocache.eq(0)
            comb += perm_attr.priv.eq(1)
            comb += perm_attr.rd_perm.eq(1)
            comb += perm_attr.wr_perm.eq(1)

        with m.If(valid_ra):
            m.d.sync += Display("DCACHE virt mode %d hit %d ra %x pte %x",
                                r0.req.virt_mode, tlb_hit.valid, ra, pte)
            m.d.sync += Display("       perm ref=%d", perm_attr.reference)
            m.d.sync += Display("       perm chg=%d", perm_attr.changed)
            m.d.sync += Display("       perm noc=%d", perm_attr.nocache)
            m.d.sync += Display("       perm prv=%d", perm_attr.priv)
            m.d.sync += Display("       perm rdp=%d", perm_attr.rd_perm)
            m.d.sync += Display("       perm wrp=%d", perm_attr.wr_perm)

    def tlb_update(self, m, r0_valid, r0, tlb_req_index,
                    tlb_hit, tlb_plru_victim):

        comb = m.d.comb
        sync = m.d.sync

        tlbie    = Signal()
        tlbwe    = Signal()

        comb += tlbie.eq(r0_valid & r0.tlbie)
        comb += tlbwe.eq(r0_valid & r0.tlbld)

        d = self.dtlb_update

        comb += d.tlbie.eq(tlbie)
        comb += d.tlbwe.eq(tlbwe)
        comb += d.doall.eq(r0.doall)
        comb += d.tlb_hit.eq(tlb_hit)
        comb += d.tlb_req_index.eq(tlb_req_index)

        with m.If(tlb_hit.valid):
            comb += d.repl_way.eq(tlb_hit.way)
        with m.Else():
            comb += d.repl_way.eq(tlb_plru_victim)
        comb += d.eatag.eq(r0.req.addr[TLB_LG_PGSZ + TLB_SET_BITS:64])
        comb += d.pte_data.eq(r0.req.data)

    def maybe_plrus(self, m, r1, plru_victim):
        """Generate PLRUs
        """
        comb = m.d.comb
        sync = m.d.sync

        if TLB_NUM_WAYS == 0:
            return

        # suite of PLRUs with a selection and output mechanism
        m.submodules.plrus = plrus = PLRUs(NUM_LINES, WAY_BITS)
        comb += plrus.way.eq(r1.hit_way)
        comb += plrus.valid.eq(r1.cache_hit)
        comb += plrus.index.eq(r1.hit_index)
        comb += plrus.isel.eq(r1.store_index) # select victim
        comb += plru_victim.eq(plrus.o_index) # selected victim

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
        sync += cache_tag_set.eq(cache_tags[index].tag)

    def dcache_request(self, m, r0, ra, req_index, req_row, req_tag,
                       r0_valid, r1, cache_tags, replace_way,
                       use_forward1_next, use_forward2_next,
                       req_hit_way, plru_victim, rc_ok, perm_attr,
                       valid_ra, perm_ok, access_ok, req_op, req_go,
                       tlb_hit, tlb_way, cache_tag_set,
                       cancel_store, req_same_tag, r0_stall, early_req_row):
        """Cache request parsing and hit detection
        """

        comb = m.d.comb
        m_in, d_in = self.m_in, self.d_in

        is_hit      = Signal()
        hit_way     = Signal(WAY_BITS)
        op          = Signal(Op)
        opsel       = Signal(3)
        go          = Signal()
        nc          = Signal()
        cache_i_validdx = Signal(NUM_WAYS)

        # Extract line, row and tag from request
        comb += req_index.eq(get_index(r0.req.addr))
        comb += req_row.eq(get_row(r0.req.addr))
        comb += req_tag.eq(get_tag(ra))

        if False: # display on comb is a bit... busy.
            comb += Display("dcache_req addr:%x ra: %x idx: %x tag: %x row: %x",
                    r0.req.addr, ra, req_index, req_tag, req_row)

        comb += go.eq(r0_valid & ~(r0.tlbie | r0.tlbld) & ~r1.ls_error)
        comb += cache_i_validdx.eq(cache_tags[req_index].valid)

        m.submodules.dcache_pend = dc = DCachePendingHit(tlb_way,
                                            cache_i_validdx, cache_tag_set,
                                            r0.req.addr)
        comb += dc.tlb_hit.eq(tlb_hit)
        comb += dc.reload_tag.eq(r1.reload_tag)
        comb += dc.virt_mode.eq(r0.req.virt_mode)
        comb += dc.go.eq(go)
        comb += dc.req_index.eq(req_index)

        comb += is_hit.eq(dc.is_hit)
        comb += hit_way.eq(dc.hit_way)
        comb += req_same_tag.eq(dc.rel_match)

        # See if the request matches the line currently being reloaded
        with m.If((r1.state == State.RELOAD_WAIT_ACK) &
                  (req_index == r1.store_index) & req_same_tag):
            # For a store, consider this a hit even if the row isn't
            # valid since it will be by the time we perform the store.
            # For a load, check the appropriate row valid bit.
            rrow = Signal(ROW_LINE_BITS)
            comb += rrow.eq(req_row)
            valid = r1.rows_valid[rrow]
            comb += is_hit.eq((~r0.req.load) | valid)
            comb += hit_way.eq(replace_way)

        # Whether to use forwarded data for a load or not
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
        with m.If((r1.forward_row1 == req_row) & (r1.forward_way1 == hit_way)):
            comb += use_forward2_next.eq(r1.forward_valid1)

        # The way that matched on a hit
        comb += req_hit_way.eq(hit_way)

        # The way to replace on a miss
        with m.If(r1.write_tag):
            comb += replace_way.eq(plru_victim)
        with m.Else():
            comb += replace_way.eq(r1.store_way)

        # work out whether we have permission for this access
        # NB we don't yet implement AMR, thus no KUAP
        comb += rc_ok.eq(perm_attr.reference
                         & (r0.req.load | perm_attr.changed))
        comb += perm_ok.eq((r0.req.priv_mode | (~perm_attr.priv)) &
                           (perm_attr.wr_perm |
                              (r0.req.load & perm_attr.rd_perm)))
        comb += access_ok.eq(valid_ra & perm_ok & rc_ok)

        # Combine the request and cache hit status to decide what
        # operation needs to be done
        comb += nc.eq(r0.req.nc | perm_attr.nocache)
        comb += op.eq(Op.OP_NONE)
        with m.If(go):
            with m.If(~access_ok):
                m.d.sync += Display("DCACHE access fail valid_ra=%d p=%d rc=%d",
                                 valid_ra, perm_ok, rc_ok)
                comb += op.eq(Op.OP_BAD)
            with m.Elif(cancel_store):
                m.d.sync += Display("DCACHE cancel store")
                comb += op.eq(Op.OP_STCX_FAIL)
            with m.Else():
                m.d.sync += Display("DCACHE valid_ra=%d nc=%d ld=%d",
                                 valid_ra, nc, r0.req.load)
                comb += opsel.eq(Cat(is_hit, nc, r0.req.load))
                with m.Switch(opsel):
                    with m.Case(0b101): comb += op.eq(Op.OP_LOAD_HIT)
                    with m.Case(0b100): comb += op.eq(Op.OP_LOAD_MISS)
                    with m.Case(0b110): comb += op.eq(Op.OP_LOAD_NC)
                    with m.Case(0b001): comb += op.eq(Op.OP_STORE_HIT)
                    with m.Case(0b000): comb += op.eq(Op.OP_STORE_MISS)
                    with m.Case(0b010): comb += op.eq(Op.OP_STORE_MISS)
                    with m.Case(0b011): comb += op.eq(Op.OP_BAD)
                    with m.Case(0b111): comb += op.eq(Op.OP_BAD)
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

        with m.If(r0_valid & r0.req.reserve):
            # XXX generate alignment interrupt if address
            # is not aligned XXX or if r0.req.nc = '1'
            with m.If(r0.req.load):
                comb += set_rsrv.eq(r0.req.atomic_last) # load with reservation
            with m.Else():
                comb += clear_rsrv.eq(r0.req.atomic_last) # store conditional
                with m.If((~reservation.valid) |
                         (r0.req.addr[LINE_OFF_BITS:64] != reservation.addr)):
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

    def writeback_control(self, m, r1, cache_out_row):
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

        comb += data_out.eq(cache_out_row)

        for i in range(8):
            with m.If(r1.forward_sel[i]):
                dsel = data_fwd.word_select(i, 8)
                comb += data_out.word_select(i, 8).eq(dsel)

        # DCache output to LoadStore
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
                sync += Display("completing load hit data=%x", data_out)

            # error cases complete without stalling
            with m.If(r1.ls_error):
                with m.If(r1.dcbz):
                    sync += Display("completing dcbz with error")
                with m.Else():
                    sync += Display("completing ld/st with error")

            # Slow ops (load miss, NC, stores)
            with m.If(r1.slow_valid):
                sync += Display("completing store or load miss adr=%x data=%x",
                                r1.req.real_addr, data_out)

        with m.Else():
            # Request came from MMU
            with m.If(r1.hit_load_valid):
                sync += Display("completing load hit to MMU, data=%x",
                                m_out.data)
            # error cases complete without stalling
            with m.If(r1.mmu_error):
                sync += Display("combpleting MMU ld with error")

            # Slow ops (i.e. load miss)
            with m.If(r1.slow_valid):
                sync += Display("completing MMU load miss, adr=%x data=%x",
                                r1.req.real_addr, m_out.data)

    def rams(self, m, r1, early_req_row, cache_out_row, replace_way):
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
        bus = self.bus

        # a Binary-to-Unary one-hots here.  replace-way one-hot is gated
        # (enabled) by bus.ack, not-write-bram, and state RELOAD_WAIT_ACK
        m.submodules.rams_replace_way_e = rwe = Decoder(NUM_WAYS)
        comb += rwe.n.eq(~((r1.state == State.RELOAD_WAIT_ACK) & bus.ack &
                   ~r1.write_bram))
        comb += rwe.i.eq(replace_way)

        m.submodules.rams_hit_way_e = hwe = Decoder(NUM_WAYS)
        comb += hwe.i.eq(r1.hit_way)

        # this one is gated with write_bram, and replace_way_e can never be
        # set at the same time.  that means that do_write can OR the outputs
        m.submodules.rams_hit_req_way_e = hre = Decoder(NUM_WAYS)
        comb += hre.n.eq(~r1.write_bram) # Decoder.n is inverted
        comb += hre.i.eq(r1.req.hit_way)

        # common Signals
        do_read  = Signal()
        wr_addr  = Signal(ROW_BITS)
        wr_data  = Signal(WB_DATA_BITS)
        wr_sel   = Signal(ROW_SIZE)
        rd_addr  = Signal(ROW_BITS)

        comb += do_read.eq(1) # always enable
        comb += rd_addr.eq(early_req_row)

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

        with m.Else():
            # Otherwise, we might be doing a reload or a DCBZ
            with m.If(r1.dcbz):
                comb += wr_data.eq(0)
            with m.Else():
                comb += wr_data.eq(bus.dat_r)
            comb += wr_addr.eq(r1.store_row)
            comb += wr_sel.eq(~0) # all 1s

        # set up Cache Rams
        for i in range(NUM_WAYS):
            do_write = Signal(name="do_wr%d" % i)
            wr_sel_m = Signal(ROW_SIZE, name="wr_sel_m_%d" % i)
            d_out   = Signal(WB_DATA_BITS, name="dout_%d" % i) # cache_row_t

            way = CacheRam(ROW_BITS, WB_DATA_BITS, ADD_BUF=True, ram_num=i)
            m.submodules["cacheram_%d" % i] = way

            comb += way.rd_en.eq(do_read)
            comb += way.rd_addr.eq(rd_addr)
            comb += d_out.eq(way.rd_data_o)
            comb += way.wr_sel.eq(wr_sel_m)
            comb += way.wr_addr.eq(wr_addr)
            comb += way.wr_data.eq(wr_data)

            # Cache hit reads
            with m.If(hwe.o[i]):
                comb += cache_out_row.eq(d_out)

            # these are mutually-exclusive via their Decoder-enablers
            # (note: Decoder-enable is inverted)
            comb += do_write.eq(hre.o[i] | rwe.o[i])

            # Mask write selects with do_write since BRAM
            # doesn't have a global write-enable
            with m.If(do_write):
                comb += wr_sel_m.eq(wr_sel)

    # Cache hit synchronous machine for the easy case.
    # This handles load hits.
    # It also handles error cases (TLB miss, cache paradox)
    def dcache_fast_hit(self, m, req_op, r0_valid, r0, r1,
                        req_hit_way, req_index, req_tag, access_ok,
                        tlb_hit, tlb_req_index):
        comb = m.d.comb
        sync = m.d.sync

        with m.If(req_op != Op.OP_NONE):
            sync += Display("op:%d addr:%x nc: %d idx: %x tag: %x way: %x",
                    req_op, r0.req.addr, r0.req.nc,
                    req_index, req_tag, req_hit_way)

        with m.If(r0_valid):
            sync += r1.mmu_req.eq(r0.mmu_req)

        # Fast path for load/store hits.
        # Set signals for the writeback controls.
        sync += r1.hit_way.eq(req_hit_way)
        sync += r1.hit_index.eq(req_index)

        sync += r1.hit_load_valid.eq(req_op == Op.OP_LOAD_HIT)
        sync += r1.cache_hit.eq((req_op == Op.OP_LOAD_HIT) |
                                (req_op == Op.OP_STORE_HIT))

        with m.If(req_op == Op.OP_BAD):
            sync += Display("Signalling ld/st error "
                            "ls_error=%i mmu_error=%i cache_paradox=%i",
                            ~r0.mmu_req,r0.mmu_req,access_ok)
            sync += r1.ls_error.eq(~r0.mmu_req)
            sync += r1.mmu_error.eq(r0.mmu_req)
            sync += r1.cache_paradox.eq(access_ok)
        with m.Else():
            sync += r1.ls_error.eq(0)
            sync += r1.mmu_error.eq(0)
            sync += r1.cache_paradox.eq(0)

        sync += r1.stcx_fail.eq(req_op == Op.OP_STCX_FAIL)

        # Record TLB hit information for updating TLB PLRU
        sync += r1.tlb_hit.eq(tlb_hit)
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
                    r0, replace_way,
                    req_hit_way, req_same_tag,
                    r0_valid, req_op, cache_tags, req_go, ra):

        comb = m.d.comb
        sync = m.d.sync
        bus = self.bus
        d_in = self.d_in

        req         = MemAccessRequest("mreq_ds")

        r1_next_cycle = Signal()
        req_row = Signal(ROW_BITS)
        req_idx = Signal(INDEX_BITS)
        req_tag = Signal(TAG_BITS)
        comb += req_idx.eq(get_index(req.real_addr))
        comb += req_row.eq(get_row(req.real_addr))
        comb += req_tag.eq(get_tag(req.real_addr))

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
                sync += r1.forward_data1.eq(bus.dat_r)
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

        with m.If((req_op == Op.OP_LOAD_HIT) | (req_op == Op.OP_STCX_FAIL)):
            with m.If(r0.mmu_req):
                sync += r1.mmu_done.eq(1)
            with m.Else():
                sync += r1.ls_valid.eq(1)

        with m.If(r1.write_tag):
            # Store new tag in selected way
            replace_way_onehot = Signal(NUM_WAYS)
            comb += replace_way_onehot.eq(1<<replace_way)
            for i in range(NUM_WAYS):
                with m.If(replace_way_onehot[i]):
                    ct = Signal(TAG_RAM_WIDTH)
                    comb += ct.eq(cache_tags[r1.store_index].tag)
                    comb += ct.word_select(i, TAG_WIDTH).eq(r1.reload_tag)
                    sync += cache_tags[r1.store_index].tag.eq(ct)
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

            with m.If(r0.req.dcbz):
                # force data to 0 for dcbz
                comb += req.data.eq(0)
            with m.Elif(r0.d_valid):
                comb += req.data.eq(r0.req.data)
            with m.Else():
                comb += req.data.eq(d_in.data)

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
                # do not let r1.state RELOAD_WAIT_ACK or STORE_WAIT_ACK
                # destroy r1.req by overwriting r1.full back to zero
                comb += r1_next_cycle.eq(1)

        # Main state machine
        with m.Switch(r1.state):

            with m.Case(State.IDLE):
                sync += r1.wb.adr.eq(req.real_addr[ROW_LINE_BITS:])
                sync += r1.wb.sel.eq(req.byte_sel)
                sync += r1.wb.dat.eq(req.data)
                sync += r1.dcbz.eq(req.dcbz)

                # Keep track of our index and way
                # for subsequent stores.
                sync += r1.store_index.eq(req_idx)
                sync += r1.store_row.eq(req_row)
                sync += r1.end_row_ix.eq(get_row_of_line(req_row)-1)
                sync += r1.reload_tag.eq(req_tag)
                sync += r1.req.same_tag.eq(1)

                with m.If(req.op == Op.OP_STORE_HIT):
                    sync += r1.store_way.eq(req.hit_way)

                #with m.If(r1.dec_acks):
                #    sync += r1.acks_pending.eq(r1.acks_pending - 1)

                # Reset per-row valid bits,
                # ready for handling OP_LOAD_MISS
                for i in range(ROW_PER_LINE):
                    sync += r1.rows_valid[i].eq(0)

                with m.If(req_op != Op.OP_NONE):
                    sync += Display("cache op %d", req.op)

                with m.Switch(req.op):
                    with m.Case(Op.OP_LOAD_HIT):
                        # stay in IDLE state
                        pass

                    with m.Case(Op.OP_LOAD_MISS):
                        sync += Display("cache miss real addr: %x " \
                                "idx: %x tag: %x",
                                req.real_addr, req_row, req_tag)

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
                            comb += r1_next_cycle.eq(0)
                            sync += r1.slow_valid.eq(1)

                            with m.If(req.mmu_req):
                                sync += r1.mmu_done.eq(1)
                            with m.Else():
                                sync += r1.ls_valid.eq(1)

                            with m.If(req.op == Op.OP_STORE_HIT):
                                sync += r1.write_bram.eq(1)
                        with m.Else():
                            # dcbz is handled much like a load miss except
                            # that we are writing to memory instead of reading
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
                ld_stbs_done = Signal()
                # Requests are all sent if stb is 0
                comb += ld_stbs_done.eq(~r1.wb.stb)

                # If we are still sending requests, was one accepted?
                with m.If((~bus.stall) & r1.wb.stb):
                    # That was the last word?  We are done sending.
                    # Clear stb and set ld_stbs_done so we can handle an
                    # eventual last ack on the same cycle.
                    # sigh - reconstruct wb adr with 3 extra 0s at front
                    wb_adr = Cat(Const(0, ROW_OFF_BITS), r1.wb.adr)
                    with m.If(is_last_row_addr(wb_adr, r1.end_row_ix)):
                        sync += r1.wb.stb.eq(0)
                        comb += ld_stbs_done.eq(1)

                    # Calculate the next row address in the current cache line
                    row = Signal(LINE_OFF_BITS-ROW_OFF_BITS)
                    comb += row.eq(r1.wb.adr)
                    sync += r1.wb.adr[:LINE_OFF_BITS-ROW_OFF_BITS].eq(row+1)

                # Incoming acks processing
                sync += r1.forward_valid1.eq(bus.ack)
                with m.If(bus.ack):
                    srow = Signal(ROW_LINE_BITS)
                    comb += srow.eq(r1.store_row)
                    sync += r1.rows_valid[srow].eq(1)

                    # If this is the data we were looking for,
                    # we can complete the request next cycle.
                    # Compare the whole address in case the
                    # request in r1.req is not the one that
                    # started this refill.
                    with m.If(req.valid & r1.req.same_tag &
                              ((r1.dcbz & r1.req.dcbz) |
                               (~r1.dcbz & (r1.req.op == Op.OP_LOAD_MISS))) &
                                (r1.store_row == get_row(req.real_addr))):
                        sync += r1.full.eq(r1_next_cycle)
                        sync += r1.slow_valid.eq(1)
                        with m.If(r1.mmu_req):
                            sync += r1.mmu_done.eq(1)
                        with m.Else():
                            sync += r1.ls_valid.eq(1)
                        sync += r1.forward_sel.eq(~0) # all 1s
                        sync += r1.use_forward1.eq(1)

                    # Check for completion
                    with m.If(ld_stbs_done & is_last_row(r1.store_row,
                                                      r1.end_row_ix)):
                        # Complete wishbone cycle
                        sync += r1.wb.cyc.eq(0)

                        # Cache line is now valid
                        cv = Signal(INDEX_BITS)
                        comb += cv.eq(cache_tags[r1.store_index].valid)
                        comb += cv.bit_select(r1.store_way, 1).eq(1)
                        sync += cache_tags[r1.store_index].valid.eq(cv)

                        sync += r1.state.eq(State.IDLE)
                        sync += Display("cache valid set %x "
                                        "idx %d way %d",
                                         cv, r1.store_index, r1.store_way)

                    # Increment store row counter
                    sync += r1.store_row.eq(next_row(r1.store_row))

            with m.Case(State.STORE_WAIT_ACK):
                st_stbs_done = Signal()
                adjust_acks = Signal(3)

                comb += st_stbs_done.eq(~r1.wb.stb)

                with m.If(r1.inc_acks != r1.dec_acks):
                    with m.If(r1.inc_acks):
                        comb += adjust_acks.eq(r1.acks_pending + 1)
                    with m.Else():
                        comb += adjust_acks.eq(r1.acks_pending - 1)
                with m.Else():
                    comb += adjust_acks.eq(r1.acks_pending)

                sync += r1.acks_pending.eq(adjust_acks)

                # Clear stb when slave accepted request
                with m.If(~bus.stall):
                    # See if there is another store waiting
                    # to be done which is in the same real page.
                    with m.If(req.valid):
                        _ra = req.real_addr[ROW_LINE_BITS:SET_SIZE_BITS]
                        sync += r1.wb.adr[0:SET_SIZE_BITS].eq(_ra)
                        sync += r1.wb.dat.eq(req.data)
                        sync += r1.wb.sel.eq(req.byte_sel)

                    with m.If((adjust_acks < 7) & req.same_tag &
                                ((req.op == Op.OP_STORE_MISS) |
                                 (req.op == Op.OP_STORE_HIT))):
                        sync += r1.wb.stb.eq(1)
                        comb += st_stbs_done.eq(0)
                        sync += r1.store_way.eq(req.hit_way)
                        sync += r1.store_row.eq(get_row(req.real_addr))

                        with m.If(req.op == Op.OP_STORE_HIT):
                            sync += r1.write_bram.eq(1)
                        sync += r1.full.eq(r1_next_cycle)
                        sync += r1.slow_valid.eq(1)

                        # Store requests never come from the MMU
                        sync += r1.ls_valid.eq(1)
                        comb += st_stbs_done.eq(0)
                        sync += r1.inc_acks.eq(1)
                    with m.Else():
                        sync += r1.wb.stb.eq(0)
                        comb += st_stbs_done.eq(1)

                # Got ack ? See if complete.
                sync += Display("got ack %d %d stbs %d adjust_acks %d",
                                bus.ack, bus.ack, st_stbs_done, adjust_acks)
                with m.If(bus.ack):
                    with m.If(st_stbs_done & (adjust_acks == 1)):
                        sync += r1.state.eq(State.IDLE)
                        sync += r1.wb.cyc.eq(0)
                        sync += r1.wb.stb.eq(0)
                    sync += r1.dec_acks.eq(1)

            with m.Case(State.NC_LOAD_WAIT_ACK):
                # Clear stb when slave accepted request
                with m.If(~bus.stall):
                    sync += r1.wb.stb.eq(0)

                # Got ack ? complete.
                with m.If(bus.ack):
                    sync += r1.state.eq(State.IDLE)
                    sync += r1.full.eq(r1_next_cycle)
                    sync += r1.slow_valid.eq(1)

                    with m.If(r1.mmu_req):
                        sync += r1.mmu_done.eq(1)
                    with m.Else():
                        sync += r1.ls_valid.eq(1)

                    sync += r1.forward_sel.eq(~0) # all 1s
                    sync += r1.use_forward1.eq(1)
                    sync += r1.wb.cyc.eq(0)
                    sync += r1.wb.stb.eq(0)

    def dcache_log(self, m, r1, valid_ra, tlb_hit, stall_out):

        sync = m.d.sync
        d_out, bus, log_out = self.d_out, self.bus, self.log_out

        sync += log_out.eq(Cat(r1.state[:3], valid_ra, tlb_hit.way[:3],
                               stall_out, req_op[:3], d_out.valid, d_out.error,
                               r1.wb.cyc, r1.wb.stb, bus.ack, bus.stall,
                               r1.real_adr[3:6]))

    def elaborate(self, platform):

        m = Module()
        comb = m.d.comb
        d_in = self.d_in

        # Storage. Hopefully "cache_rows" is a BRAM, the rest is LUTs
        cache_tags       = CacheTagArray()
        cache_tag_set    = Signal(TAG_RAM_WIDTH)

        # TODO attribute ram_style : string;
        # TODO attribute ram_style of cache_tags : signal is "distributed";

        """note: these are passed to nmigen.hdl.Memory as "attributes".
           don't know how, just that they are.
        """
        # TODO attribute ram_style of
        #  dtlb_tags : signal is "distributed";
        # TODO attribute ram_style of
        #  dtlb_ptes : signal is "distributed";

        r0      = RegStage0("r0")
        r0_full = Signal()

        r1 = RegStage1("r1")

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

        cache_out_row     = Signal(WB_DATA_BITS)

        plru_victim       = Signal(WAY_BITS)
        replace_way       = Signal(WAY_BITS)

        # Wishbone read/write/cache write formatting signals
        bus_sel           = Signal(8)

        # TLB signals
        tlb_way       = TLBRecord("tlb_way")
        tlb_req_index = Signal(TLB_SET_BITS)
        tlb_hit       = TLBHit("tlb_hit")
        pte           = Signal(TLB_PTE_BITS)
        ra            = Signal(REAL_ADDR_BITS)
        valid_ra      = Signal()
        perm_attr     = PermAttr("dc_perms")
        rc_ok         = Signal()
        perm_ok       = Signal()
        access_ok     = Signal()

        tlb_plru_victim = Signal(TLB_WAY_BITS)

        # we don't yet handle collisions between loadstore1 requests
        # and MMU requests
        comb += self.m_out.stall.eq(0)

        # Hold off the request in r0 when r1 has an uncompleted request
        comb += r0_stall.eq(r0_full & (r1.full | d_in.hold))
        comb += r0_valid.eq(r0_full & ~r1.full & ~d_in.hold)
        comb += self.stall_out.eq(r0_stall)

        # deal with litex not doing wishbone pipeline mode
        # XXX in wrong way.  FIFOs are needed in the SRAM test
        # so that stb/ack match up. same thing done in icache.py
        if not self.microwatt_compat:
            comb += self.bus.stall.eq(self.bus.cyc & ~self.bus.ack)

        # Wire up wishbone request latch out of stage 1
        comb += self.bus.we.eq(r1.wb.we)
        comb += self.bus.adr.eq(r1.wb.adr)
        comb += self.bus.sel.eq(r1.wb.sel)
        comb += self.bus.stb.eq(r1.wb.stb)
        comb += self.bus.dat_w.eq(r1.wb.dat)
        comb += self.bus.cyc.eq(r1.wb.cyc)

        # create submodule TLBUpdate
        m.submodules.dtlb_update = self.dtlb_update = DTLBUpdate()

        # call sub-functions putting everything together, using shared
        # signals established above
        self.stage_0(m, r0, r1, r0_full)
        self.tlb_read(m, r0_stall, tlb_way)
        self.tlb_search(m, tlb_req_index, r0, r0_valid,
                        tlb_way,
                        pte, tlb_hit, valid_ra, perm_attr, ra)
        self.tlb_update(m, r0_valid, r0, tlb_req_index,
                        tlb_hit, tlb_plru_victim)
        self.maybe_plrus(m, r1, plru_victim)
        self.maybe_tlb_plrus(m, r1, tlb_plru_victim, tlb_req_index)
        self.cache_tag_read(m, r0_stall, req_index, cache_tag_set, cache_tags)
        self.dcache_request(m, r0, ra, req_index, req_row, req_tag,
                           r0_valid, r1, cache_tags, replace_way,
                           use_forward1_next, use_forward2_next,
                           req_hit_way, plru_victim, rc_ok, perm_attr,
                           valid_ra, perm_ok, access_ok, req_op, req_go,
                           tlb_hit, tlb_way, cache_tag_set,
                           cancel_store, req_same_tag, r0_stall, early_req_row)
        self.reservation_comb(m, cancel_store, set_rsrv, clear_rsrv,
                           r0_valid, r0, reservation)
        self.reservation_reg(m, r0_valid, access_ok, set_rsrv, clear_rsrv,
                           reservation, r0)
        self.writeback_control(m, r1, cache_out_row)
        self.rams(m, r1, early_req_row, cache_out_row, replace_way)
        self.dcache_fast_hit(m, req_op, r0_valid, r0, r1,
                        req_hit_way, req_index, req_tag, access_ok,
                        tlb_hit, tlb_req_index)
        self.dcache_slow(m, r1, use_forward1_next, use_forward2_next,
                    r0, replace_way,
                    req_hit_way, req_same_tag,
                         r0_valid, req_op, cache_tags, req_go, ra)
        #self.dcache_log(m, r1, valid_ra, tlb_hit, stall_out)

        return m


if __name__ == '__main__':
    dut = DCache()
    vl = rtlil.convert(dut, ports=[])
    with open("test_dcache.il", "w") as f:
        f.write(vl)
