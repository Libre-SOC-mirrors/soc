"""ICache

based on Anton Blanchard microwatt icache.vhdl

Set associative icache

TODO (in no specific order):
* Add debug interface to inspect cache content
* Add snoop/invalidate path
* Add multi-hit error detection
* Pipelined bus interface (wb or axi)
* Maybe add parity? There's a few bits free in each BRAM row on Xilinx
* Add optimization: service hits on partially loaded lines
* Add optimization: (maybe) interrupt reload on fluch/redirect
* Check if playing with the geometry of the cache tags allow for more
  efficient use of distributed RAM and less logic/muxes. Currently we
  write TAG_BITS width which may not match full ram blocks and might
  cause muxes to be inferred for "partial writes".
* Check if making the read size of PLRU a ROM helps utilization

Links:

* https://bugs.libre-soc.org/show_bug.cgi?id=485
* https://libre-soc.org/irclog-microwatt/%23microwatt.2021-12-07.log.html
  (discussion about brams for ECP5)

"""

from enum import (Enum, unique)
from nmigen import (Module, Signal, Elaboratable, Cat, Array, Const, Repl,
                    Record)
from nmigen.cli import main, rtlil
from nmutil.iocontrol import RecordObject
from nmigen.utils import log2_int
from nmigen.lib.coding import Decoder
from nmutil.util import Display
from nmutil.latch import SRLatch

#from nmutil.plru import PLRU
from soc.experiment.plru import PLRU, PLRUs
from soc.experiment.cache_ram import CacheRam

from soc.experiment.mem_types import (Fetch1ToICacheType,
                                      ICacheToDecode1Type,
                                      MMUToICacheType)

from soc.experiment.wb_types import (WB_ADDR_BITS, WB_DATA_BITS,
                                     WB_SEL_BITS, WBAddrType, WBDataType,
                                     WBSelType, WBMasterOut, WBSlaveOut,
                                     )

from nmigen_soc.wishbone.bus import Interface
from soc.minerva.units.fetch import FetchUnitInterface


# for test
from soc.bus.sram import SRAM
from nmigen import Memory
from nmutil.util import wrap
from nmigen.cli import main, rtlil

# NOTE: to use cxxsim, export NMIGEN_SIM_MODE=cxxsim from the shell
# Also, check out the cxxsim nmigen branch, and latest yosys from git
from nmutil.sim_tmp_alternative import Simulator, Settle

# from microwatt/utils.vhdl
def ispow2(n):
    return n != 0 and (n & (n - 1)) == 0

SIM            = 0
# Non-zero to enable log data collection
LOG_LENGTH     = 0

class ICacheConfig:
    def __init__(self, LINE_SIZE     = 64,
                       NUM_LINES     = 64,  # Number of lines in a set
                       NUM_WAYS      = 2,  # Number of ways
                       TLB_SIZE      = 64,  # L1 ITLB number of entries
                       TLB_LG_PGSZ   = 12): # L1 ITLB log_2(page_size)
        self.LINE_SIZE      = LINE_SIZE
        self.NUM_LINES      = NUM_LINES
        self.NUM_WAYS       = NUM_WAYS
        self.TLB_SIZE       = TLB_SIZE
        self.TLB_LG_PGSZ    = TLB_LG_PGSZ

        # BRAM organisation: We never access more than wishbone_data_bits
        # at a time so to save resources we make the array only that wide,
        # and use consecutive indices for to make a cache "line"
        #
        # self.ROW_SIZE is the width in bytes of the BRAM
        # (based on WB, so 64-bits)
        self.ROW_SIZE       = WB_DATA_BITS // 8
        # Number of real address bits that we store
        self.REAL_ADDR_BITS = 56

        self.ROW_SIZE_BITS  = self.ROW_SIZE * 8
        # ROW_PER_LINE is the number of row (wishbone) transactions in a line
        self.ROW_PER_LINE   = self.LINE_SIZE // self.ROW_SIZE
        # BRAM_ROWS is the number of rows in BRAM
        # needed to represent the full icache
        self.BRAM_ROWS      = self.NUM_LINES * self.ROW_PER_LINE
        # INSN_PER_ROW is the number of 32bit instructions per BRAM row
        self.INSN_PER_ROW   = self.ROW_SIZE_BITS // 32

        # Bit fields counts in the address
        #
        # INSN_BITS is the number of bits to select an instruction in a row
        self.INSN_BITS      = log2_int(self.INSN_PER_ROW)
        # ROW_BITS is the number of bits to select a row
        self.ROW_BITS       = log2_int(self.BRAM_ROWS)
        # ROW_LINE_BITS is the number of bits to select a row within a line
        self.ROW_LINE_BITS  = log2_int(self.ROW_PER_LINE)
        # LINE_OFF_BITS is the number of bits for the offset in a cache line
        self.LINE_OFF_BITS  = log2_int(self.LINE_SIZE)
        # ROW_OFF_BITS is the number of bits for the offset in a row
        self.ROW_OFF_BITS   = log2_int(self.ROW_SIZE)
        # INDEX_BITS is the number of bits to select a cache line
        self.INDEX_BITS     = log2_int(self.NUM_LINES)
        # SET_SIZE_BITS is the log base 2 of the set size
        self.SET_SIZE_BITS  = self.LINE_OFF_BITS + self.INDEX_BITS
        # TAG_BITS is the number of bits of the tag part of the address
        self.TAG_BITS       = self.REAL_ADDR_BITS - self.SET_SIZE_BITS
        # TAG_WIDTH is the width in bits of each way of the tag RAM
        self.TAG_WIDTH      = self.TAG_BITS + 7 - ((self.TAG_BITS + 7) % 8)

        # WAY_BITS is the number of bits to select a way
        self.WAY_BITS       = log2_int(self.NUM_WAYS)
        self.TAG_RAM_WIDTH  = self.TAG_BITS * self.NUM_WAYS

        # L1 ITLB
        self.TL_BITS        = log2_int(self.TLB_SIZE)
        self.TLB_EA_TAG_BITS = 64 - (self.TLB_LG_PGSZ + self.TL_BITS)
        self.TLB_PTE_BITS    = 64

        print("self.BRAM_ROWS       =", self.BRAM_ROWS)
        print("self.INDEX_BITS      =", self.INDEX_BITS)
        print("self.INSN_BITS       =", self.INSN_BITS)
        print("self.INSN_PER_ROW    =", self.INSN_PER_ROW)
        print("self.LINE_SIZE       =", self.LINE_SIZE)
        print("self.LINE_OFF_BITS   =", self.LINE_OFF_BITS)
        print("LOG_LENGTH      =", LOG_LENGTH)
        print("self.NUM_LINES       =", self.NUM_LINES)
        print("self.NUM_WAYS        =", self.NUM_WAYS)
        print("self.REAL_ADDR_BITS  =", self.REAL_ADDR_BITS)
        print("self.ROW_BITS        =", self.ROW_BITS)
        print("self.ROW_OFF_BITS    =", self.ROW_OFF_BITS)
        print("self.ROW_LINE_BITS   =", self.ROW_LINE_BITS)
        print("self.ROW_PER_LINE    =", self.ROW_PER_LINE)
        print("self.ROW_SIZE        =", self.ROW_SIZE)
        print("self.ROW_SIZE_BITS   =", self.ROW_SIZE_BITS)
        print("self.SET_SIZE_BITS   =", self.SET_SIZE_BITS)
        print("SIM             =", SIM)
        print("self.TAG_BITS        =", self.TAG_BITS)
        print("self.TAG_RAM_WIDTH   =", self.TAG_RAM_WIDTH)
        print("self.TAG_BITS        =", self.TAG_BITS)
        print("self.TL_BITS        =", self.TL_BITS)
        print("self.TLB_EA_TAG_BITS =", self.TLB_EA_TAG_BITS)
        print("self.TLB_LG_PGSZ     =", self.TLB_LG_PGSZ)
        print("self.TLB_PTE_BITS    =", self.TLB_PTE_BITS)
        print("self.TLB_SIZE        =", self.TLB_SIZE)
        print("self.WAY_BITS        =", self.WAY_BITS)

        assert self.LINE_SIZE % self.ROW_SIZE == 0
        assert ispow2(self.LINE_SIZE), "self.LINE_SIZE not power of 2"
        assert ispow2(self.NUM_LINES), "self.NUM_LINES not power of 2"
        assert ispow2(self.ROW_PER_LINE), "self.ROW_PER_LINE not power of 2"
        assert ispow2(self.INSN_PER_ROW), "self.INSN_PER_ROW not power of 2"
        assert (self.ROW_BITS == (self.INDEX_BITS + self.ROW_LINE_BITS)), \
            "geometry bits don't add up"
        assert (self.LINE_OFF_BITS ==
            (self.ROW_OFF_BITS + self.ROW_LINE_BITS)), \
           "geometry bits don't add up"
        assert (self.REAL_ADDR_BITS ==
            (self.TAG_BITS + self.INDEX_BITS + self.LINE_OFF_BITS)), \
            "geometry bits don't add up"
        assert (self.REAL_ADDR_BITS ==
            (self.TAG_BITS + self.ROW_BITS + self.ROW_OFF_BITS)), \
            "geometry bits don't add up"

        # Example of layout for 32 lines of 64 bytes:
        #
        # ..  tag    |index|  line  |
        # ..         |   row   |    |
        # ..         |     |   | |00| zero          (2)
        # ..         |     |   |-|  | self.INSN_BITS     (1)
        # ..         |     |---|    | self.ROW_LINE_BITS  (3)
        # ..         |     |--- - --| self.LINE_OFF_BITS (6)
        # ..         |         |- --| self.ROW_OFF_BITS  (3)
        # ..         |----- ---|    | self.ROW_BITS      (8)
        # ..         |-----|        | self.INDEX_BITS    (5)
        # .. --------|              | self.TAG_BITS      (53)

    # The cache data BRAM organized as described above for each way
    #subtype cache_row_t is std_ulogic_vector(self.ROW_SIZE_BITS-1 downto 0);
    #
    def RowPerLineValidArray(self):
        return Array(Signal(name="rows_valid_%d" %x) \
                     for x in range(self.ROW_PER_LINE))


    # TODO to be passed to nigmen as ram attributes
    # attribute ram_style : string;
    # attribute ram_style of cache_tags : signal is "distributed";

    def TLBRecord(self, name):
        tlb_layout = [ ('tag', self.TLB_EA_TAG_BITS),
                      ('pte', self.TLB_PTE_BITS)
                     ]
        return Record(tlb_layout, name=name)

    def TLBArray(self):
        return Array(self.TLBRecord("tlb%d" % x) for x in range(self.TLB_SIZE))

    # PLRU output interface
    def PLRUOut(self):
        return Array(Signal(self.WAY_BITS, name="plru_out_%d" %x) \
                     for x in range(self.NUM_LINES))

    # Return the cache line index (tag index) for an address
    def get_index(self, addr):
        return addr[self.LINE_OFF_BITS:self.SET_SIZE_BITS]

    # Return the cache row index (data memory) for an address
    def get_row(self, addr):
        return addr[self.ROW_OFF_BITS:self.SET_SIZE_BITS]

    # Return the index of a row within a line
    def get_row_of_line(self, row):
        return row[:self.ROW_BITS][:self.ROW_LINE_BITS]

    # Returns whether this is the last row of a line
    def is_last_row_addr(self, addr, last):
        return addr[self.ROW_OFF_BITS:self.LINE_OFF_BITS] == last

    # Returns whether this is the last row of a line
    def is_last_row(self, row, last):
        return self.get_row_of_line(row) == last

    # Return the next row in the current cache line. We use a dedicated
    # function in order to limit the size of the generated adder to be
    # only the bits within a cache line (3 bits with default settings)
    def next_row(self, row):
        row_v = row[0:self.ROW_LINE_BITS] + 1
        return Cat(row_v[:self.ROW_LINE_BITS], row[self.ROW_LINE_BITS:])

    # Read the instruction word for the given address
    # in the current cache row
    def read_insn_word(self, addr, data):
        word = addr[2:self.INSN_BITS+2]
        return data.word_select(word, 32)

    # Get the tag value from the address
    def get_tag(self, addr):
        return addr[self.SET_SIZE_BITS:self.REAL_ADDR_BITS]

    # Read a tag from a tag memory row
    def read_tag(self, way, tagset):
        return tagset.word_select(way, self.TAG_BITS)

    # Write a tag to tag memory row
    def write_tag(self, way, tagset, tag):
        return self.read_tag(way, tagset).eq(tag)

    # Simple hash for direct-mapped TLB index
    def hash_ea(self, addr):
        hsh = (addr[self.TLB_LG_PGSZ:self.TLB_LG_PGSZ + self.TL_BITS] ^
               addr[self.TLB_LG_PGSZ + self.TL_BITS:
                    self.TLB_LG_PGSZ + 2 * self.TL_BITS ] ^
               addr[self.TLB_LG_PGSZ + 2 * self.TL_BITS:
                    self.TLB_LG_PGSZ + 3 * self.TL_BITS])
        return hsh


# Cache reload state machine
@unique
class State(Enum):
    IDLE     = 0
    CLR_TAG  = 1
    WAIT_ACK = 2


class RegInternal(RecordObject):
    def __init__(self, cfg):
        super().__init__()
        # Cache hit state (Latches for 1 cycle BRAM access)
        self.hit_way      = Signal(cfg.WAY_BITS)
        self.hit_nia      = Signal(64)
        self.hit_smark    = Signal()
        self.hit_valid    = Signal()

        # Cache miss state (reload state machine)
        self.state        = Signal(State, reset=State.IDLE)
        self.wb           = WBMasterOut("wb")
        self.req_adr      = Signal(64)
        self.store_way    = Signal(cfg.WAY_BITS)
        self.store_index  = Signal(cfg.INDEX_BITS)
        self.store_row    = Signal(cfg.ROW_BITS)
        self.store_tag    = Signal(cfg.TAG_BITS)
        self.store_valid  = Signal()
        self.end_row_ix   = Signal(cfg.ROW_LINE_BITS)
        self.rows_valid   = cfg.RowPerLineValidArray()

        # TLB miss state
        self.fetch_failed = Signal()


class ICache(FetchUnitInterface, Elaboratable, ICacheConfig):
    """64 bit direct mapped icache. All instructions are 4B aligned."""
    def __init__(self, pspec):
        FetchUnitInterface.__init__(self, pspec)
        self.i_in           = Fetch1ToICacheType(name="i_in")
        self.i_out          = ICacheToDecode1Type(name="i_out")

        self.m_in           = MMUToICacheType(name="m_in")

        self.stall_in       = Signal()
        self.stall_out      = Signal()
        self.flush_in       = Signal()
        self.inval_in       = Signal()

        # standard naming (wired to non-standard for compatibility)
        self.bus = Interface(addr_width=32,
                            data_width=64,
                            granularity=8,
                            features={'stall'},
                            #alignment=0,
                            name="icache_wb")

        self.log_out        = Signal(54)

        # use FetchUnitInterface, helps keep some unit tests running
        self.use_fetch_iface = False

        # test if microwatt compatibility is to be enabled
        self.microwatt_compat = (hasattr(pspec, "microwatt_compat") and
                                 (pspec.microwatt_compat == True))

        if self.microwatt_compat:
            # reduce way sizes and num lines
            ICacheConfig.__init__(self, NUM_LINES = 4,
                                        NUM_WAYS = 1,
                                 )
        else:
            ICacheConfig.__init__(self)

    def use_fetch_interface(self):
        self.use_fetch_iface = True

    # Generate a cache RAM for each way
    def rams(self, m, r, cache_out_row, use_previous,
             replace_way, req_row):

        comb = m.d.comb
        sync = m.d.sync

        bus, stall_in = self.bus, self.stall_in

        # read condition (for every cache ram)
        do_read  = Signal()
        comb += do_read.eq(~(stall_in | use_previous))

        rd_addr  = Signal(self.ROW_BITS)
        wr_addr  = Signal(self.ROW_BITS)
        comb += rd_addr.eq(req_row)
        comb += wr_addr.eq(r.store_row)

        # binary-to-unary converters: replace-way enabled by bus.ack,
        # hit-way left permanently enabled
        m.submodules.replace_way_e = re = Decoder(self.NUM_WAYS)
        m.submodules.hit_way_e = he = Decoder(self.NUM_WAYS)
        comb += re.i.eq(replace_way)
        comb += re.n.eq(~bus.ack)
        comb += he.i.eq(r.hit_way)

        for i in range(self.NUM_WAYS):
            do_write = Signal(name="do_wr_%d" % i)
            d_out    = Signal(self.ROW_SIZE_BITS, name="d_out_%d" % i)
            wr_sel   = Signal(self.ROW_SIZE, name="wr_sel_%d" % i)

            way = CacheRam(self.ROW_BITS, self.ROW_SIZE_BITS,
                           TRACE=True, ram_num=i)
            m.submodules["cacheram_%d" % i] =  way

            comb += way.rd_en.eq(do_read)
            comb += way.rd_addr.eq(rd_addr)
            comb += d_out.eq(way.rd_data_o)
            comb += way.wr_sel.eq(wr_sel)
            comb += way.wr_addr.eq(wr_addr)
            comb += way.wr_data.eq(bus.dat_r)

            comb += do_write.eq(re.o[i])

            with m.If(do_write):
                sync += Display("cache write adr: %x data: %lx",
                                wr_addr, way.wr_data)

            with m.If(he.o[i]):
                comb += cache_out_row.eq(d_out)
                with m.If(do_read):
                    sync += Display("cache read adr: %x data: %x",
                                     req_row, d_out)

            comb += wr_sel.eq(Repl(do_write, self.ROW_SIZE))

    # Generate PLRUs
    def maybe_plrus(self, m, r, plru_victim):
        comb = m.d.comb

        if self.NUM_WAYS == 0:
            return


        m.submodules.plrus = plru = PLRUs(self.NUM_LINES, self.WAY_BITS)
        comb += plru.way.eq(r.hit_way)
        comb += plru.valid.eq(r.hit_valid)
        comb += plru.index.eq(self.get_index(r.hit_nia))
        comb += plru.isel.eq(r.store_index) # select victim
        comb += plru_victim.eq(plru.o_index) # selected victim

    # TLB hit detection and real address generation
    def itlb_lookup(self, m, tlb_req_index, itlb, itlb_valid,
                    real_addr, ra_valid, eaa_priv,
                    priv_fault, access_ok):

        comb = m.d.comb

        i_in = self.i_in

        # use an *asynchronous* Memory read port here (combinatorial)
        m.submodules.rd_tlb = rd_tlb = self.tlbmem.read_port(domain="comb")
        tlb = self.TLBRecord("tlb_rdport")
        pte, ttag = tlb.pte, tlb.tag

        comb += tlb_req_index.eq(self.hash_ea(i_in.nia))
        comb += rd_tlb.addr.eq(tlb_req_index)
        comb += tlb.eq(rd_tlb.data)

        with m.If(i_in.virt_mode):
            comb += real_addr.eq(Cat(i_in.nia[:self.TLB_LG_PGSZ],
                                     pte[self.TLB_LG_PGSZ:self.REAL_ADDR_BITS]))

            with m.If(ttag == i_in.nia[self.TLB_LG_PGSZ + self.TL_BITS:64]):
                comb += ra_valid.eq(itlb_valid.q.bit_select(tlb_req_index, 1))

            comb += eaa_priv.eq(pte[3])

        with m.Else():
            comb += real_addr.eq(i_in.nia[:self.REAL_ADDR_BITS])
            comb += ra_valid.eq(1)
            comb += eaa_priv.eq(1)

        # No IAMR, so no KUEP support for now
        comb += priv_fault.eq(eaa_priv & ~i_in.priv_mode)
        comb += access_ok.eq(ra_valid & ~priv_fault)

    # iTLB update
    def itlb_update(self, m, itlb, itlb_valid):
        comb = m.d.comb
        sync = m.d.sync

        m_in = self.m_in

        wr_index = Signal(self.TL_BITS)
        wr_unary = Signal(self.TLB_SIZE)
        comb += wr_index.eq(self.hash_ea(m_in.addr))
        comb += wr_unary.eq(1<<wr_index)

        m.submodules.wr_tlb = wr_tlb = self.tlbmem.write_port()
        sync += itlb_valid.s.eq(0)
        sync += itlb_valid.r.eq(0)

        with m.If(m_in.tlbie & m_in.doall):
            # Clear all valid bits
            sync += itlb_valid.r.eq(-1)

        with m.Elif(m_in.tlbie):
            # Clear entry regardless of hit or miss
            sync += itlb_valid.r.eq(wr_unary)

        with m.Elif(m_in.tlbld):
            tlb = self.TLBRecord("tlb_wrport")
            comb += tlb.tag.eq(m_in.addr[self.TLB_LG_PGSZ + self.TL_BITS:64])
            comb += tlb.pte.eq(m_in.pte)
            comb += wr_tlb.en.eq(1)
            comb += wr_tlb.addr.eq(wr_index)
            comb += wr_tlb.data.eq(tlb)
            sync += itlb_valid.s.eq(wr_unary)

    # Cache hit detection, output to fetch2 and other misc logic
    def icache_comb(self, m, use_previous, r, req_index, req_row,
                    req_hit_way, req_tag, real_addr, req_laddr,
                    cache_valids, access_ok,
                    req_is_hit, req_is_miss, replace_way,
                    plru_victim, cache_out_row):

        comb = m.d.comb
        m.submodules.rd_tag = rd_tag = self.tagmem.read_port(domain="comb")

        i_in, i_out, bus = self.i_in, self.i_out, self.bus
        flush_in, stall_out = self.flush_in, self.stall_out

        is_hit  = Signal()
        hit_way = Signal(self.WAY_BITS)

        # i_in.sequential means that i_in.nia this cycle is 4 more than
        # last cycle.  If we read more than 32 bits at a time, had a
        # cache hit last cycle, and we don't want the first 32-bit chunk
        # then we can keep the data we read last cycle and just use that.
        with m.If(i_in.nia[2:self.INSN_BITS+2] != 0):
            comb += use_previous.eq(i_in.sequential & r.hit_valid)

        # Extract line, row and tag from request
        comb += req_index.eq(self.get_index(i_in.nia))
        comb += req_row.eq(self.get_row(i_in.nia))
        comb += req_tag.eq(self.get_tag(real_addr))

        # Calculate address of beginning of cache row, will be
        # used for cache miss processing if needed
        comb += req_laddr.eq(Cat(
                 Const(0, self.ROW_OFF_BITS),
                 real_addr[self.ROW_OFF_BITS:self.REAL_ADDR_BITS],
                ))

        # Test if pending request is a hit on any way
        hitcond = Signal()
        comb += hitcond.eq((r.state == State.WAIT_ACK)
                 & (req_index == r.store_index)
                 & r.rows_valid[req_row % self.ROW_PER_LINE]
                )
        # i_in.req asserts Decoder active
        cvb = Signal(self.NUM_WAYS)
        ctag = Signal(self.TAG_RAM_WIDTH)
        comb += rd_tag.addr.eq(req_index)
        comb += ctag.eq(rd_tag.data)
        comb += cvb.eq(cache_valids.q.word_select(req_index, self.NUM_WAYS))
        m.submodules.store_way_e = se = Decoder(self.NUM_WAYS)
        comb += se.i.eq(r.store_way)
        comb += se.n.eq(~i_in.req)
        for i in range(self.NUM_WAYS):
            tagi = Signal(self.TAG_BITS, name="tag_i%d" % i)
            hit_test = Signal(name="hit_test%d" % i)
            is_tag_hit = Signal(name="is_tag_hit_%d" % i)
            comb += tagi.eq(self.read_tag(i, ctag))
            comb += hit_test.eq(se.o[i])
            comb += is_tag_hit.eq((cvb[i] | (hitcond & hit_test)) &
                                  (tagi == req_tag))
            with m.If(is_tag_hit):
                comb += hit_way.eq(i)
                comb += is_hit.eq(1)

        # Generate the "hit" and "miss" signals
        # for the synchronous blocks
        with m.If(i_in.req & access_ok & ~flush_in):
            comb += req_is_hit.eq(is_hit)
            comb += req_is_miss.eq(~is_hit)

        comb += req_hit_way.eq(hit_way)

        # The way to replace on a miss
        with m.If(r.state == State.CLR_TAG):
            comb += replace_way.eq(plru_victim)
        with m.Else():
            comb += replace_way.eq(r.store_way)

        # Output instruction from current cache row
        #
        # Note: This is a mild violation of our design principle of
        # having pipeline stages output from a clean latch. In this
        # case we output the result of a mux. The alternative would
        # be output an entire row which I prefer not to do just yet
        # as it would force fetch2 to know about some of the cache
        # geometry information.
        comb += i_out.insn.eq(self.read_insn_word(r.hit_nia, cache_out_row))
        comb += i_out.valid.eq(r.hit_valid)
        comb += i_out.nia.eq(r.hit_nia)
        comb += i_out.stop_mark.eq(r.hit_smark)
        comb += i_out.fetch_failed.eq(r.fetch_failed)

        # Stall fetch1 if we have a miss on cache or TLB
        # or a protection fault
        comb += stall_out.eq(~(is_hit & access_ok))

        # Wishbone requests output (from the cache miss reload machine)
        comb += bus.we.eq(r.wb.we)
        comb += bus.adr.eq(r.wb.adr)
        comb += bus.sel.eq(r.wb.sel)
        comb += bus.stb.eq(r.wb.stb)
        comb += bus.dat_w.eq(r.wb.dat)
        comb += bus.cyc.eq(r.wb.cyc)

    # Cache hit synchronous machine
    def icache_hit(self, m, use_previous, r, req_is_hit, req_hit_way,
                   req_index, req_tag, real_addr):
        sync = m.d.sync

        i_in, stall_in = self.i_in, self.stall_in
        flush_in       = self.flush_in

        # keep outputs to fetch2 unchanged on a stall
        # except that flush or reset sets valid to 0
        # If use_previous, keep the same data as last
        # cycle and use the second half
        with m.If(stall_in | use_previous):
            with m.If(flush_in):
                sync += r.hit_valid.eq(0)
        with m.Else():
            # On a hit, latch the request for the next cycle,
            # when the BRAM data will be available on the
            # cache_out output of the corresponding way
            sync += r.hit_valid.eq(req_is_hit)

            with m.If(req_is_hit):
                sync += r.hit_way.eq(req_hit_way)
                sync += Display("cache hit nia:%x IR:%x SM:%x idx:%x tag:%x "
                                "way:%x RA:%x", i_in.nia, i_in.virt_mode,
                                 i_in.stop_mark, req_index, req_tag,
                                 req_hit_way, real_addr)

        with m.If(~stall_in):
            # Send stop marks and NIA down regardless of validity
            sync += r.hit_smark.eq(i_in.stop_mark)
            sync += r.hit_nia.eq(i_in.nia)

    def icache_miss_idle(self, m, r, req_is_miss, req_laddr,
                         req_index, req_tag, replace_way, real_addr):
        comb = m.d.comb
        sync = m.d.sync

        i_in = self.i_in

        # Reset per-row valid flags, only used in WAIT_ACK
        for i in range(self.ROW_PER_LINE):
            sync += r.rows_valid[i].eq(0)

        # We need to read a cache line
        with m.If(req_is_miss):
            sync += Display(
                     "cache miss nia:%x IR:%x SM:%x idx:%x "
                     " way:%x tag:%x RA:%x", i_in.nia,
                     i_in.virt_mode, i_in.stop_mark, req_index,
                     replace_way, req_tag, real_addr)

            # Keep track of our index and way for subsequent stores
            st_row = Signal(self.ROW_BITS)
            comb += st_row.eq(self.get_row(req_laddr))
            sync += r.store_index.eq(req_index)
            sync += r.store_row.eq(st_row)
            sync += r.store_tag.eq(req_tag)
            sync += r.store_valid.eq(1)
            sync += r.end_row_ix.eq(self.get_row_of_line(st_row) - 1)

            # Prep for first wishbone read.  We calculate the address
            # of the start of the cache line and start the WB cycle.
            sync += r.req_adr.eq(req_laddr)
            sync += r.wb.cyc.eq(1)
            sync += r.wb.stb.eq(1)

            # Track that we had one request sent
            sync += r.state.eq(State.CLR_TAG)

    def icache_miss_clr_tag(self, m, r, replace_way,
                            req_index,
                            cache_valids):
        comb = m.d.comb
        sync = m.d.sync
        m.submodules.wr_tag = wr_tag = self.tagmem.write_port(
                                                    granularity=self.TAG_BITS)

        # Get victim way from plru
        sync += r.store_way.eq(replace_way)

        # Force misses on that way while reloading that line
        idx = req_index*self.NUM_WAYS + replace_way # 2D index, 1st dim: self.NUM_WAYS
        comb += cache_valids.r.eq(1<<idx)

        # use write-port "granularity" to select the tag to write to
        # TODO: the Memory should be multipled-up (by NUM_TAGS)
        tagset = Signal(self.TAG_RAM_WIDTH)
        comb += tagset.eq(r.store_tag << (replace_way*self.TAG_BITS))
        comb += wr_tag.en.eq(1<<replace_way)
        comb += wr_tag.addr.eq(r.store_index)
        comb += wr_tag.data.eq(tagset)

        sync += r.state.eq(State.WAIT_ACK)

    def icache_miss_wait_ack(self, m, r, replace_way, inval_in,
                             cache_valids, stbs_done):
        comb = m.d.comb
        sync = m.d.sync

        bus = self.bus

        # Requests are all sent if stb is 0
        stbs_zero = Signal()
        comb += stbs_zero.eq(r.wb.stb == 0)
        comb += stbs_done.eq(stbs_zero)

        # If we are still sending requests, was one accepted?
        with m.If(~bus.stall & ~stbs_zero):
            # That was the last word? We are done sending.
            # Clear stb and set stbs_done so we can handle
            # an eventual last ack on the same cycle.
            with m.If(self.is_last_row_addr(r.req_adr, r.end_row_ix)):
                sync += Display("IS_LAST_ROW_ADDR r.wb.addr:%x "
                         "r.end_row_ix:%x r.wb.stb:%x stbs_zero:%x "
                         "stbs_done:%x", r.wb.adr, r.end_row_ix,
                         r.wb.stb, stbs_zero, stbs_done)
                sync += r.wb.stb.eq(0)
                comb += stbs_done.eq(1)

            # Calculate the next row address
            rarange = Signal(self.LINE_OFF_BITS - self.ROW_OFF_BITS)
            comb += rarange.eq(r.req_adr[self.ROW_OFF_BITS:
                                         self.LINE_OFF_BITS] + 1)
            sync += r.req_adr[self.ROW_OFF_BITS:self.LINE_OFF_BITS].eq(rarange)
            sync += Display("RARANGE r.req_adr:%x rarange:%x "
                            "stbs_zero:%x stbs_done:%x",
                            r.req_adr, rarange, stbs_zero, stbs_done)

        # Incoming acks processing
        with m.If(bus.ack):
            sync += Display("WB_IN_ACK data:%x stbs_zero:%x "
                            "stbs_done:%x",
                            bus.dat_r, stbs_zero, stbs_done)

            sync += r.rows_valid[r.store_row % self.ROW_PER_LINE].eq(1)

            # Check for completion
            with m.If(stbs_done & self.is_last_row(r.store_row, r.end_row_ix)):
                # Complete wishbone cycle
                sync += r.wb.cyc.eq(0)
                # be nice, clear addr
                sync += r.req_adr.eq(0)

                # Cache line is now valid
                idx = r.store_index*self.NUM_WAYS + replace_way # 2D index again
                valid = r.store_valid & ~inval_in
                comb += cache_valids.s.eq(1<<idx)
                sync += r.state.eq(State.IDLE)

            # move on to next request in row
            # Increment store row counter
            sync += r.store_row.eq(self.next_row(r.store_row))

    # Cache miss/reload synchronous machine
    def icache_miss(self, m, r, req_is_miss,
                    req_index, req_laddr, req_tag, replace_way,
                    cache_valids, access_ok, real_addr):
        comb = m.d.comb
        sync = m.d.sync

        i_in, bus, m_in  = self.i_in, self.bus, self.m_in
        stall_in, flush_in = self.stall_in, self.flush_in
        inval_in           = self.inval_in

        stbs_done = Signal()

        comb += r.wb.sel.eq(-1)
        comb += r.wb.adr.eq(r.req_adr[3:])

        # Process cache invalidations
        with m.If(inval_in):
            comb += cache_valids.r.eq(-1)
            sync += r.store_valid.eq(0)

        # Main state machine
        with m.Switch(r.state):

            with m.Case(State.IDLE):
                self.icache_miss_idle(m, r, req_is_miss, req_laddr,
                                      req_index, req_tag, replace_way,
                                      real_addr)

            with m.Case(State.CLR_TAG, State.WAIT_ACK):
                with m.If(r.state == State.CLR_TAG):
                    self.icache_miss_clr_tag(m, r, replace_way,
                                             req_index,
                                             cache_valids)

                self.icache_miss_wait_ack(m, r, replace_way, inval_in,
                                          cache_valids, stbs_done)

        # TLB miss and protection fault processing
        with m.If(flush_in | m_in.tlbld):
            sync += r.fetch_failed.eq(0)
        with m.Elif(i_in.req & ~access_ok & ~stall_in):
            sync += r.fetch_failed.eq(1)

    # icache_log: if LOG_LENGTH > 0 generate
    def icache_log(self, m, req_hit_way, ra_valid, access_ok,
                   req_is_miss, req_is_hit, lway, wstate, r):
        comb = m.d.comb
        sync = m.d.sync

        bus, i_out       = self.bus, self.i_out
        log_out, stall_out = self.log_out, self.stall_out

        # Output data to logger
        for i in range(LOG_LENGTH):
            log_data = Signal(54)
            lway     = Signal(self.WAY_BITS)
            wstate   = Signal()

            sync += lway.eq(req_hit_way)
            sync += wstate.eq(0)

            with m.If(r.state != State.IDLE):
                sync += wstate.eq(1)

            sync += log_data.eq(Cat(
                     ra_valid, access_ok, req_is_miss, req_is_hit,
                     lway, wstate, r.hit_nia[2:6], r.fetch_failed,
                     stall_out, bus.stall, r.wb.cyc, r.wb.stb,
                     r.real_addr[3:6], bus.ack, i_out.insn, i_out.valid
                    ))
            comb += log_out.eq(log_data)

    def elaborate(self, platform):

        m                = Module()
        comb             = m.d.comb

        # Cache-Ways "valid" indicators.  this is a 2D Signal, by the
        # number of ways and the number of lines.
        vec = SRLatch(sync=True, llen=self.NUM_WAYS*self.NUM_LINES,
                      name="cachevalids")
        m.submodules.cache_valids = cache_valids = vec

        # TLB Array
        itlb            = self.TLBArray()
        vec = SRLatch(sync=False, llen=self.TLB_SIZE, name="tlbvalids")
        m.submodules.itlb_valids = itlb_valid = vec

        # TODO to be passed to nmigen as ram attributes
        # attribute ram_style of itlb_tags : signal is "distributed";
        # attribute ram_style of itlb_ptes : signal is "distributed";

        # Privilege bit from PTE EAA field
        eaa_priv         = Signal()

        r                = RegInternal(self)

        # Async signal on incoming request
        req_index        = Signal(self.INDEX_BITS)
        req_row          = Signal(self.ROW_BITS)
        req_hit_way      = Signal(self.WAY_BITS)
        req_tag          = Signal(self.TAG_BITS)
        req_is_hit       = Signal()
        req_is_miss      = Signal()
        req_laddr        = Signal(64)

        tlb_req_index    = Signal(self.TL_BITS)
        real_addr        = Signal(self.REAL_ADDR_BITS)
        ra_valid         = Signal()
        priv_fault       = Signal()
        access_ok        = Signal()
        use_previous     = Signal()

        cache_out_row    = Signal(self.ROW_SIZE_BITS)

        plru_victim      = Signal(self.WAY_BITS)
        replace_way      = Signal(self.WAY_BITS)

        self.tlbmem = Memory(depth=self.TLB_SIZE,
                             width=self.TLB_EA_TAG_BITS+self.TLB_PTE_BITS)
        self.tagmem = Memory(depth=self.NUM_LINES,
                             width=self.TAG_RAM_WIDTH)

        # call sub-functions putting everything together,
        # using shared signals established above
        self.rams(m, r, cache_out_row, use_previous, replace_way, req_row)
        self.maybe_plrus(m, r, plru_victim)
        self.itlb_lookup(m, tlb_req_index, itlb, itlb_valid, real_addr,
                         ra_valid, eaa_priv, priv_fault,
                         access_ok)
        self.itlb_update(m, itlb, itlb_valid)
        self.icache_comb(m, use_previous, r, req_index, req_row, req_hit_way,
                         req_tag, real_addr, req_laddr,
                         cache_valids,
                         access_ok, req_is_hit, req_is_miss,
                         replace_way, plru_victim, cache_out_row)
        self.icache_hit(m, use_previous, r, req_is_hit, req_hit_way,
                        req_index, req_tag, real_addr)
        self.icache_miss(m, r, req_is_miss, req_index,
                         req_laddr, req_tag, replace_way,
                         cache_valids,
                         access_ok, real_addr)
        #self.icache_log(m, log_out, req_hit_way, ra_valid, access_ok,
        #                req_is_miss, req_is_hit, lway, wstate, r)

        # don't connect up to FetchUnitInterface so that some unit tests
        # can continue to operate
        if not self.use_fetch_iface:
            return m

        # connect to FetchUnitInterface. FetchUnitInterface is undocumented
        # so needs checking and iterative revising
        i_in, bus, i_out = self.i_in, self.bus, self.i_out
        comb += i_in.req.eq(self.a_i_valid)
        comb += i_in.nia.eq(self.a_pc_i)
        comb += self.stall_in.eq(self.a_stall_i)
        comb += self.f_fetch_err_o.eq(i_out.fetch_failed)
        comb += self.f_badaddr_o.eq(i_out.nia)
        comb += self.f_instr_o.eq(i_out.insn)
        comb += self.f_busy_o.eq(~i_out.valid) # probably

        # TODO, connect dcache wb_in/wb_out to "standard" nmigen Wishbone bus
        ibus = self.ibus
        comb += ibus.adr.eq(self.bus.adr)
        comb += ibus.dat_w.eq(self.bus.dat_w)
        comb += ibus.sel.eq(self.bus.sel)
        comb += ibus.cyc.eq(self.bus.cyc)
        comb += ibus.stb.eq(self.bus.stb)
        comb += ibus.we.eq(self.bus.we)

        comb += self.bus.dat_r.eq(ibus.dat_r)
        comb += self.bus.ack.eq(ibus.ack)
        if hasattr(ibus, "stall"):
            comb += self.bus.stall.eq(ibus.stall)
        else:
            # fake-up the wishbone stall signal to comply with pipeline mode
            # same thing is done in dcache.py
            comb += self.bus.stall.eq(self.bus.cyc & ~self.bus.ack)

        return m


def icache_sim(dut):
    i_in = dut.i_in
    i_out  = dut.i_out
    m_out = dut.m_in

    yield i_in.priv_mode.eq(1)
    yield i_in.req.eq(0)
    yield i_in.nia.eq(0)
    yield i_in.stop_mark.eq(0)
    yield m_out.tlbld.eq(0)
    yield m_out.tlbie.eq(0)
    yield m_out.addr.eq(0)
    yield m_out.pte.eq(0)
    yield
    yield
    yield
    yield

    # miss, stalls for a bit
    yield i_in.req.eq(1)
    yield i_in.nia.eq(Const(0x0000000000000004, 64))
    yield
    valid = yield i_out.valid
    while not valid:
        yield
        valid = yield i_out.valid
    yield i_in.req.eq(0)

    insn  = yield i_out.insn
    nia   = yield i_out.nia
    assert insn == 0x00000001, \
        "insn @%x=%x expected 00000001" % (nia, insn)
    yield i_in.req.eq(0)
    yield

    # hit
    yield i_in.req.eq(1)
    yield i_in.nia.eq(Const(0x0000000000000008, 64))
    yield
    valid = yield i_out.valid
    while not valid:
        yield
        valid = yield i_out.valid
    yield i_in.req.eq(0)

    nia   = yield i_out.nia
    insn  = yield i_out.insn
    yield
    assert insn == 0x00000002, \
        "insn @%x=%x expected 00000002" % (nia, insn)

    # another miss
    yield i_in.req.eq(1)
    yield i_in.nia.eq(Const(0x0000000000000040, 64))
    yield
    valid = yield i_out.valid
    while not valid:
        yield
        valid = yield i_out.valid
    yield i_in.req.eq(0)

    nia   = yield i_in.nia
    insn  = yield i_out.insn
    assert insn == 0x00000010, \
        "insn @%x=%x expected 00000010" % (nia, insn)

    # test something that aliases (this only works because
    # the unit test SRAM is a depth of 512)
    yield i_in.req.eq(1)
    yield i_in.nia.eq(Const(0x0000000000000100, 64))
    yield
    yield
    valid = yield i_out.valid
    assert ~valid
    for i in range(30):
        yield
    yield
    insn  = yield i_out.insn
    valid = yield i_out.valid
    insn  = yield i_out.insn
    assert valid
    assert insn == 0x00000040, \
         "insn @%x=%x expected 00000040" % (nia, insn)
    yield i_in.req.eq(0)


def test_icache(mem):
    from soc.config.test.test_loadstore import TestMemPspec
    pspec = TestMemPspec(addr_wid=32,
                         mask_wid=8,
                         reg_wid=64,
                         )
    dut    = ICache(pspec)

    memory = Memory(width=64, depth=512, init=mem)
    sram   = SRAM(memory=memory, granularity=8)

    m      = Module()

    m.submodules.icache = dut
    m.submodules.sram   = sram

    m.d.comb += sram.bus.cyc.eq(dut.bus.cyc)
    m.d.comb += sram.bus.stb.eq(dut.bus.stb)
    m.d.comb += sram.bus.we.eq(dut.bus.we)
    m.d.comb += sram.bus.sel.eq(dut.bus.sel)
    m.d.comb += sram.bus.adr.eq(dut.bus.adr)
    m.d.comb += sram.bus.dat_w.eq(dut.bus.dat_w)

    m.d.comb += dut.bus.ack.eq(sram.bus.ack)
    m.d.comb += dut.bus.dat_r.eq(sram.bus.dat_r)

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    sim.add_sync_process(wrap(icache_sim(dut)))
    with sim.write_vcd('test_icache.vcd'):
         sim.run()


if __name__ == '__main__':
    from soc.config.test.test_loadstore import TestMemPspec
    pspec = TestMemPspec(addr_wid=64,
                         mask_wid=8,
                         reg_wid=64,
                         )
    dut = ICache(pspec)
    vl = rtlil.convert(dut, ports=[])
    with open("test_icache.il", "w") as f:
        f.write(vl)

    # set up memory every 32-bits with incrementing values 0 1 2 ...
    mem = []
    for i in range(512):
        mem.append((i*2) | ((i*2+1)<<32))

    test_icache(mem)
