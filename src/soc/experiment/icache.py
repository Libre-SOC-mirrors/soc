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

"""
from enum import Enum, unique
from nmigen import (Module, Signal, Elaboratable, Cat, Signal)
from nmigen.cli import main
from nmigen.cli import rtlil
from nmutil.iocontrol import RecordObject
from nmutil.byterev import byte_reverse
from nmutil.mask import Mask
from nmigen.util import log2_int


from soc.experiment.mem_types import Fetch1ToICacheType,
                                     ICacheToDecode1Type,
                                     MMUToICacheType

from experiment.wb_types import WB_ADDR_BITS, WB_DATA_BITS, WB_SEL_BITS,
                                WBAddrType, WBDataType, WBSelType,
                                WbMasterOut, WBSlaveOut,
                                WBMasterOutVector, WBSlaveOutVector,
                                WBIOMasterOut, WBIOSlaveOut


# Cache reload state machine
@unique
class State(Enum):
    IDLE
    CLR_TAG
    WAIT_ACK

#     type reg_internal_t is record
# 	-- Cache hit state (Latches for 1 cycle BRAM access)
# 	hit_way   : way_t;
# 	hit_nia   : std_ulogic_vector(63 downto 0);
# 	hit_smark : std_ulogic;
# 	hit_valid : std_ulogic;
#
# 	-- Cache miss state (reload state machine)
#         state            : state_t;
#         wb               : wishbone_master_out;
# 	store_way        : way_t;
#         store_index      : index_t;
# 	store_row        : row_t;
#         store_tag        : cache_tag_t;
#         store_valid      : std_ulogic;
#         end_row_ix       : row_in_line_t;
#         rows_valid       : row_per_line_valid_t;
#
#         -- TLB miss state
#         fetch_failed     : std_ulogic;
#     end record;
class RegInternal(RecordObject):
    def __init__(self):
        super().__init__()
        # Cache hit state (Latches for 1 cycle BRAM access)
        self.hit_way      = Signal(NUM_WAYS)
        self.hit_nia      = Signal(64)
        self.hit_smark    = Signal()
        self.hit_valid    = Signal()

        # Cache miss state (reload state machine)
        self.state        = State()
        self.wb           = WBMasterOut()
        self.store_way    = Signal(NUM_WAYS)
        self.store_index  = Signal(NUM_LINES)
        self.store_row    = Signal(BRAM_ROWS)
        self.store_tag    = Signal(TAG_BITS)
        self.store_valid  = Signal()
        self.end_row_ix   = Signal(ROW_LINE_BITS)
        self.rows_valid   = RowPerLineValidArray()

        # TLB miss state
        self.fetch_failed = Signal()

# -- 64 bit direct mapped icache. All instructions are 4B aligned.
#
# entity icache is
#     generic (
#         SIM : boolean := false;
#         -- Line size in bytes
#         LINE_SIZE : positive := 64;
#         -- BRAM organisation: We never access more than wishbone_data_bits
#         -- at a time so to save resources we make the array only that wide,
#         -- and use consecutive indices for to make a cache "line"
#         --
#         -- ROW_SIZE is the width in bytes of the BRAM (based on WB,
#         -- so 64-bits)
#         ROW_SIZE  : positive := wishbone_data_bits / 8;
#         -- Number of lines in a set
#         NUM_LINES : positive := 32;
#         -- Number of ways
#         NUM_WAYS  : positive := 4;
#         -- L1 ITLB number of entries (direct mapped)
#         TLB_SIZE : positive := 64;
#         -- L1 ITLB log_2(page_size)
#         TLB_LG_PGSZ : positive := 12;
#         -- Number of real address bits that we store
#         REAL_ADDR_BITS : positive := 56;
#         -- Non-zero to enable log data collection
#         LOG_LENGTH : natural := 0
#         );
#     port (
#         clk          : in std_ulogic;
#         rst          : in std_ulogic;
#
#         i_in         : in Fetch1ToIcacheType;
#         i_out        : out IcacheToDecode1Type;
#
#         m_in         : in MmuToIcacheType;
#
#         stall_in     : in std_ulogic;
# 	stall_out    : out std_ulogic;
# 	flush_in     : in std_ulogic;
# 	inval_in     : in std_ulogic;
#
#         wishbone_out : out wishbone_master_out;
#         wishbone_in  : in wishbone_slave_out;
#
#         log_out      : out std_ulogic_vector(53 downto 0)
#         );
# end entity icache;
# 64 bit direct mapped icache. All instructions are 4B aligned.
class ICache(Elaboratable):
    """64 bit direct mapped icache. All instructions are 4B aligned."""
    def __init__(self):
        self.SIM            = 0
        self.LINE_SIZE      = 64
        # BRAM organisation: We never access more than wishbone_data_bits
        # at a time so to save resources we make the array only that wide,
        # and use consecutive indices for to make a cache "line"
        #
        # ROW_SIZE is the width in bytes of the BRAM (based on WB, so 64-bits)
        self.ROW_SIZE       = WB_DATA_BITS / 8
        # Number of lines in a set
        self.NUM_LINES      = 32
        # Number of ways
        self.NUM_WAYS       = 4
        # L1 ITLB number of entries (direct mapped)
        self.TLB_SIZE       = 64
        # L1 ITLB log_2(page_size)
        self.TLB_LG_PGSZ    = 12
        # Number of real address bits that we store
        self.REAL_ADDR_BITS = 56
        # Non-zero to enable log data collection
        self.LOG_LENGTH     = 0

        self.i_in           = Fetch1ToICacheType()
        self.i_out          = ICacheToDecode1Type()

        self.m_in           = MMUToICacheType()

        self.stall_in       = Signal()
        self.stall_out      = Signal()
        self.flush_in       = Signal()
        self.inval_in       = Signal()

        self.wb_out         = WBMasterOut()
        self.wb_in          = WBSlaveOut()

        self.log_out        = Signal(54)

#     -- Return the cache line index (tag index) for an address
#     function get_index(addr: std_ulogic_vector(63 downto 0))
#      return index_t is
#     begin
#         return to_integer(unsigned(
#          addr(SET_SIZE_BITS - 1 downto LINE_OFF_BITS)
#         ));
#     end;
    # Return the cache line index (tag index) for an address
    def get_index(addr):
        return addr[LINE_OFF_BITS:SET_SIZE_BITS]

#     -- Return the cache row index (data memory) for an address
#     function get_row(addr: std_ulogic_vector(63 downto 0)) return row_t is
#     begin
#         return to_integer(unsigned(
#          addr(SET_SIZE_BITS - 1 downto ROW_OFF_BITS)
#         ));
#     end;
    # Return the cache row index (data memory) for an address
    def get_row(addr):
        return addr[ROW_OFF_BITS:SET_SIZE_BITS]

#     -- Return the index of a row within a line
#     function get_row_of_line(row: row_t) return row_in_line_t is
# 	variable row_v : unsigned(ROW_BITS-1 downto 0);
#     begin
# 	row_v := to_unsigned(row, ROW_BITS);
#         return row_v(ROW_LINEBITS-1 downto 0);
#     end;
    # Return the index of a row within a line
    def get_row_of_line(row):
        row[:ROW_LINE_BITS]

#     -- Returns whether this is the last row of a line
#     function is_last_row_addr(addr: wishbone_addr_type; last: row_in_line_t)
#      return boolean is
#     begin
# 	return unsigned(addr(LINE_OFF_BITS-1 downto ROW_OFF_BITS)) = last;
#     end;
    # Returns whether this is the last row of a line
    def is_last_row_addr(addr, last):
        return addr[ROW_OFF_BITS:LINE_OFF_BITS] == last

#     -- Returns whether this is the last row of a line
#     function is_last_row(row: row_t; last: row_in_line_t) return boolean is
#     begin
# 	return get_row_of_line(row) = last;
#     end;
    # Returns whether this is the last row of a line
    def is_last_row(row, last):
        return get_row_of_line(row) == last

#     -- Return the address of the next row in the current cache line
#     function next_row_addr(addr: wishbone_addr_type)
# 	return std_ulogic_vector is
# 	variable row_idx : std_ulogic_vector(ROW_LINEBITS-1 downto 0);
# 	variable result  : wishbone_addr_type;
#     begin
# 	-- Is there no simpler way in VHDL to generate that 3 bits adder ?
# 	row_idx := addr(LINE_OFF_BITS-1 downto ROW_OFF_BITS);
# 	row_idx := std_ulogic_vector(unsigned(row_idx) + 1);
# 	result := addr;
# 	result(LINE_OFF_BITS-1 downto ROW_OFF_BITS) := row_idx;
# 	return result;
#     end;
    # Return the address of the next row in the current cache line
    def next_row_addr(addr):
        # TODO no idea what's going on here, looks like double assignments
        # overriding earlier assignments ??? Help please!

#     -- Return the next row in the current cache line. We use a dedicated
#     -- function in order to limit the size of the generated adder to be
#     -- only the bits within a cache line (3 bits with default settings)
#     function next_row(row: row_t) return row_t is
# 	variable row_v   : std_ulogic_vector(ROW_BITS-1 downto 0);
# 	variable row_idx : std_ulogic_vector(ROW_LINEBITS-1 downto 0);
# 	variable result  : std_ulogic_vector(ROW_BITS-1 downto 0);
#     begin
# 	row_v := std_ulogic_vector(to_unsigned(row, ROW_BITS));
# 	row_idx := row_v(ROW_LINEBITS-1 downto 0);
# 	row_v(ROW_LINEBITS-1 downto 0) :=
#        std_ulogic_vector(unsigned(row_idx) + 1);
# 	return to_integer(unsigned(row_v));
#     end;
    # Return the next row in the current cache line. We use a dedicated
    # function in order to limit the size of the generated adder to be
    # only the bits within a cache line (3 bits with default settings)
    def next_row(row):
        # TODO no idea what's going on here, looks like double assignments
        # overriding earlier assignments ??? Help please!

#     -- Read the instruction word for the given address in the
#     -- current cache row
#     function read_insn_word(addr: std_ulogic_vector(63 downto 0);
# 			    data: cache_row_t) return std_ulogic_vector is
# 	variable word: integer range 0 to INSN_PER_ROW-1;
#     begin
#         word := to_integer(unsigned(addr(INSN_BITS+2-1 downto 2)));
# 	return data(31+word*32 downto word*32);
#     end;
    # Read the instruction word for the given address
    # in the current cache row
    def read_insn_word(addr, data):
        word = addr[2:INSN_BITS+3]
        return data[word * 32:32 + word * 32]

#     -- Get the tag value from the address
#     function get_tag(addr: std_ulogic_vector(REAL_ADDR_BITS - 1 downto 0))
#      return cache_tag_t is
#     begin
#         return addr(REAL_ADDR_BITS - 1 downto SET_SIZE_BITS);
#     end;
    # Get the tag value from the address
    def get_tag(addr):
        return addr[SET_SIZE_BITS:REAL_ADDR_BITS]

#     -- Read a tag from a tag memory row
#     function read_tag(way: way_t; tagset: cache_tags_set_t)
#      return cache_tag_t is
#     begin
# 	return tagset((way+1) * TAG_BITS - 1 downto way * TAG_BITS);
#     end;
    # Read a tag from a tag memory row
    def read_tag(way, tagset):
        return tagset[way * TAG_BITS:(way + 1) * TAG_BITS]

#     -- Write a tag to tag memory row
#     procedure write_tag(way: in way_t; tagset: inout cache_tags_set_t;
# 			tag: cache_tag_t) is
#     begin
# 	tagset((way+1) * TAG_BITS - 1 downto way * TAG_BITS) := tag;
#     end;
    # Write a tag to tag memory row
    def write_tag(way, tagset, tag):
        tagset[way * TAG_BITS:(way + 1) * TAG_BITS] = tag

#     -- Simple hash for direct-mapped TLB index
#     function hash_ea(addr: std_ulogic_vector(63 downto 0))
#      return tlb_index_t is
#         variable hash : std_ulogic_vector(TLB_BITS - 1 downto 0);
#     begin
#         hash := addr(TLB_LG_PGSZ + TLB_BITS - 1 downto TLB_LG_PGSZ)
#                 xor addr(
#                  TLB_LG_PGSZ + 2 * TLB_BITS - 1 downto
#                  TLB_LG_PGSZ + TLB_BITS
#                 )
#                 xor addr(
#                  TLB_LG_PGSZ + 3 * TLB_BITS - 1 downto
#                  TLB_LG_PGSZ + 2 * TLB_BITS
#                 );
#         return to_integer(unsigned(hash));
#     end;
    # Simple hash for direct-mapped TLB index
    def hash_ea(addr):
        hsh = addr[TLB_LG_PGSZ:TLB_LG_PGSZ + TLB_BITS]
               ^ addr[TLB_LG_PGSZ + TLB_BITS:TLB_LG_PGSZ + 2 * TLB_BITS]
               ^ addr[TLB_LG_PGSZ + 2 * TLB_BITS: TLB_LG_PGSZ + 3 * TLB_BITS]
        return hsh

#     -- Generate a cache RAM for each way
#     rams: for i in 0 to NUM_WAYS-1 generate
# 	signal do_read  : std_ulogic;
# 	signal do_write : std_ulogic;
# 	signal rd_addr  : std_ulogic_vector(ROW_BITS-1 downto 0);
# 	signal wr_addr  : std_ulogic_vector(ROW_BITS-1 downto 0);
# 	signal dout     : cache_row_t;
# 	signal wr_sel   : std_ulogic_vector(ROW_SIZE-1 downto 0);
#     begin
# 	way: entity work.cache_ram
# 	    generic map (
# 		ROW_BITS => ROW_BITS,
# 		WIDTH => ROW_SIZE_BITS
# 		)
# 	    port map (
# 		clk     => clk,
# 		rd_en   => do_read,
# 		rd_addr => rd_addr,
# 		rd_data => dout,
# 		wr_sel  => wr_sel,
# 		wr_addr => wr_addr,
# 		wr_data => wishbone_in.dat
# 		);
# 	process(all)
# 	begin
# 	    do_read <= not (stall_in or use_previous);
# 	    do_write <= '0';
# 	    if wishbone_in.ack = '1' and replace_way = i then
# 		do_write <= '1';
# 	    end if;
# 	    cache_out(i) <= dout;
# 	    rd_addr <= std_ulogic_vector(to_unsigned(req_row, ROW_BITS));
# 	    wr_addr <= std_ulogic_vector(to_unsigned(r.store_row, ROW_BITS));
#             for i in 0 to ROW_SIZE-1 loop
#                 wr_sel(i) <= do_write;
#             end loop;
# 	end process;
#     end generate;
    def rams(self, m):
        comb = m.d.comb

        do_read  = Signal()
        do_write = Signal()
        rd_addr  = Signal(ROW_BITS)
        wr_addr  = Signal(ROW_BITS)
        _d_out   = Signal(ROW_SIZE_BITS)
        wr_sel   = Signal(ROW_SIZE)

        for i in range(NUM_WAYS)
            way = CacheRam(ROW_BITS, ROW_SIZE_BITS)
            comb += way.rd_en.eq(do_read)
            comb += way.rd_addr.eq(rd_addr)
            comb += way.rd_data.eq(_d_out)
            comb += way.wr_sel.eq(wr_sel)
            comb += way.wr_add.eq(wr_addr)
            comb += way.wr_data.eq(wb_in.dat)

            comb += do_read.eq(~(stall_in | use_previous))
            comb += do_write.eq(0)

            with m.If(wb_in.ack & (replace_way == i)):
                do_write.eq(1)

            comb += cache_out[i].eq(_d_out)
            comb += rd_addr.eq(Signal(req_row))
            comb += wr_addr.eq(Signal(r.store_row))
            for j in range(ROW_SIZE):
                comb += wr_sel[j].eq(do_write)

#     -- Generate PLRUs
#     maybe_plrus: if NUM_WAYS > 1 generate
#     begin
# 	plrus: for i in 0 to NUM_LINES-1 generate
# 	    -- PLRU interface
# 	    signal plru_acc    : std_ulogic_vector(WAY_BITS-1 downto 0);
# 	    signal plru_acc_en : std_ulogic;
# 	    signal plru_out    : std_ulogic_vector(WAY_BITS-1 downto 0);
#
# 	begin
# 	    plru : entity work.plru
# 		generic map (
# 		    BITS => WAY_BITS
# 		    )
# 		port map (
# 		    clk => clk,
# 		    rst => rst,
# 		    acc => plru_acc,
# 		    acc_en => plru_acc_en,
# 		    lru => plru_out
# 		    );
#
# 	    process(all)
# 	    begin
# 		-- PLRU interface
# 		if get_index(r.hit_nia) = i then
# 		    plru_acc_en <= r.hit_valid;
# 		else
# 		    plru_acc_en <= '0';
# 		end if;
# 		plru_acc <=
#                std_ulogic_vector(to_unsigned(r.hit_way, WAY_BITS));
# 		plru_victim(i) <= plru_out;
# 	    end process;
# 	end generate;
#     end generate;
    def maybe_plrus(self, m):
        comb += m.d.comb

        with m.If(NUM_WAYS > 1):
            for i in range(NUM_LINES):
                plru_acc    = Signal(WAY_BITS)
                plru_acc_en = Signal()
                plru_out    = Signal(WAY_BITS)
                plru        = PLRU(WAY_BITS)
                comb += plru.acc.eq(plru_acc)
                comb += plru.acc_en.eq(plru_acc_en)
                comb += plru.lru.eq(plru_out)

                # PLRU interface
                with m.If(get_index(r.hit_nia) == i):
                    comb += plru.acc_en.eq(r.hit_valid)

                with m.Else():
                    comb += plru.acc_en.eq(0)

                comb += plru.acc.eq(r.hit_way)
                comb += plru_victim[i].eq(plru.lru)

#     -- TLB hit detection and real address generation
#     itlb_lookup : process(all)
#         variable pte : tlb_pte_t;
#         variable ttag : tlb_tag_t;
#     begin
#         tlb_req_index <= hash_ea(i_in.nia);
#         pte := itlb_ptes(tlb_req_index);
#         ttag := itlb_tags(tlb_req_index);
#         if i_in.virt_mode = '1' then
#             real_addr <= pte(REAL_ADDR_BITS - 1 downto TLB_LG_PGSZ) &
#                          i_in.nia(TLB_LG_PGSZ - 1 downto 0);
#             if ttag = i_in.nia(63 downto TLB_LG_PGSZ + TLB_BITS) then
#                 ra_valid <= itlb_valids(tlb_req_index);
#             else
#                 ra_valid <= '0';
#             end if;
#             eaa_priv <= pte(3);
#         else
#             real_addr <= i_in.nia(REAL_ADDR_BITS - 1 downto 0);
#             ra_valid <= '1';
#             eaa_priv <= '1';
#         end if;
#
#         -- no IAMR, so no KUEP support for now
#         priv_fault <= eaa_priv and not i_in.priv_mode;
#         access_ok <= ra_valid and not priv_fault;
#     end process;
    # TLB hit detection and real address generation
    def itlb_lookup(self, m):
        comb = m.d.comb

        comb += tlb_req_index.eq(hash_ea(i_in.nia))
        comb += pte.eq(itlb_ptes[tlb_req_index])
        comb += ttag.eq(itlb_tags[tlb_req_index])

        with m.If(i_in.virt_mode):
            comb += real_addr.eq(Cat(
                     i_in.nia[:TLB_LB_PGSZ],
                     pte[TLB_LG_PGSZ:REAL_ADDR_BITS]
                    ))

            with m.If(ttag == i_in.nia[TLB_LG_PGSZ + TLB_BITS:64]):
                comb += ra_valid.eq(itlb_valid_bits[tlb_req_index])

            with m.Else():
                comb += ra_valid.eq(0)

        with m.Else():
            comb += real_addr.eq(i_in.nia[:REAL_ADDR_BITS])
            comb += ra_valid.eq(1)
            comb += eaa_priv.eq(1)

        # No IAMR, so no KUEP support for now
        comb += priv_fault.eq(eaa_priv & ~i_in.priv_mode)
        comb += access_ok.eq(ra_valid & ~priv_fault)

#     -- iTLB update
#     itlb_update: process(clk)
#         variable wr_index : tlb_index_t;
#     begin
#         if rising_edge(clk) then
#             wr_index := hash_ea(m_in.addr);
#             if rst = '1' or (m_in.tlbie = '1' and m_in.doall = '1') then
#                 -- clear all valid bits
#                 for i in tlb_index_t loop
#                     itlb_valids(i) <= '0';
#                 end loop;
#             elsif m_in.tlbie = '1' then
#                 -- clear entry regardless of hit or miss
#                 itlb_valids(wr_index) <= '0';
#             elsif m_in.tlbld = '1' then
#                 itlb_tags(wr_index) <= m_in.addr(
#                                         63 downto TLB_LG_PGSZ + TLB_BITS
#                                        );
#                 itlb_ptes(wr_index) <= m_in.pte;
#                 itlb_valids(wr_index) <= '1';
#             end if;
#         end if;
#     end process;
    # iTLB update
    def itlb_update(self, m):
        sync = m.d.sync

        wr_index = Signal(TLB_SIZE)
        sync += wr_index.eq(hash_ea(m_in.addr))

        with m.If('''TODO rst in nmigen''' | (m_in.tlbie & m_in.doall)):
            # Clear all valid bits
            for i in range(TLB_SIZE):
                sync += itlb_vlaids[i].eq(0)

        with m.Elif(m_in.tlbie):
            # Clear entry regardless of hit or miss
            sync += itlb_valid_bits[wr_index].eq(0)

        with m.Elif(m_in.tlbld):
            sync += itlb_tags[wr_index].eq(
                     m_in.addr[TLB_LG_PGSZ + TLB_BITS:64]
                    )
            sync += itlb_ptes[wr_index].eq(m_in.pte)
            sync += itlb_valid_bits[wr_index].eq(1)

#     -- Cache hit detection, output to fetch2 and other misc logic
#     icache_comb : process(all)
    # Cache hit detection, output to fetch2 and other misc logic
    def icache_comb(self, m):
# 	variable is_hit  : std_ulogic;
# 	variable hit_way : way_t;
        comb = m.d.comb

        is_hit  = Signal()
        hit_way = Signal(NUM_WAYS)
#     begin
#         -- i_in.sequential means that i_in.nia this cycle is 4 more than
#         -- last cycle.  If we read more than 32 bits at a time, had a
#         -- cache hit last cycle, and we don't want the first 32-bit chunk
#         -- then we can keep the data we read last cycle and just use that.
#         if unsigned(i_in.nia(INSN_BITS+2-1 downto 2)) /= 0 then
#             use_previous <= i_in.sequential and r.hit_valid;
#         else
#             use_previous <= '0';
#         end if;
        # i_in.sequential means that i_in.nia this cycle is 4 more than
        # last cycle.  If we read more than 32 bits at a time, had a
        # cache hit last cycle, and we don't want the first 32-bit chunk
        # then we can keep the data we read last cycle and just use that.
        with m.If(i_in.nia[2:INSN_BITS+2] != 0):
            comb += use_previous.eq(i_in.sequential & r.hit_valid)

        with m.else():
            comb += use_previous.eq(0)

# 	-- Extract line, row and tag from request
#         req_index <= get_index(i_in.nia);
#         req_row <= get_row(i_in.nia);
#         req_tag <= get_tag(real_addr);
        # Extract line, row and tag from request
        comb += req_index.eq(get_index(i_in.nia))
        comb += req_row.eq(get_row(i_in.nia))
        comb += req_tag.eq(get_tag(real_addr))

# 	-- Calculate address of beginning of cache row, will be
# 	-- used for cache miss processing if needed
# 	req_laddr <= (63 downto REAL_ADDR_BITS => '0') &
#                      real_addr(REAL_ADDR_BITS - 1 downto ROW_OFF_BITS) &
# 		     (ROW_OFF_BITS-1 downto 0 => '0');
        # Calculate address of beginning of cache row, will be
        # used for cache miss processing if needed
        comb += req_laddr.eq(Cat(
                 Const(0b0, ROW_OFF_BITS),
                 real_addr[ROW_OFF_BITS:REAL_ADDR_BITS],
                 Const(0, REAL_ADDR_BITS)
                ))

# 	-- Test if pending request is a hit on any way
# 	hit_way := 0;
# 	is_hit := '0';
# 	for i in way_t loop
# 	    if i_in.req = '1' and
#                 (cache_valids(req_index)(i) = '1' or
#                  (r.state = WAIT_ACK and
#                   req_index = r.store_index and
#                   i = r.store_way and
#                   r.rows_valid(req_row mod ROW_PER_LINE) = '1')) then
# 		if read_tag(i, cache_tags(req_index)) = req_tag then
# 		    hit_way := i;
# 		    is_hit := '1';
# 		end if;
# 	    end if;
# 	end loop;
        # Test if pending request is a hit on any way
        for i in range(NUM_WAYS):
            with m.If(i_in.req &
                      (cache_valid_bits[req_index][i] |
                       ((r.state == State.WAIT_ACK)
                        & (req_index == r.store_index)
                        & (i == r.store_way)
                        & r.rows_valid[req_row % ROW_PER_LINE])):
                with m.If(read_tag(i, cahce_tags[req_index]) == req_tag):
                    comb += hit_way.eq(i)
                    comb += is_hit.eq(1)

# 	-- Generate the "hit" and "miss" signals for the synchronous blocks
#       if i_in.req = '1' and access_ok = '1' and flush_in = '0'
#        and rst = '0' then
#           req_is_hit  <= is_hit;
#           req_is_miss <= not is_hit;
#       else
#           req_is_hit  <= '0';
#           req_is_miss <= '0';
#       end if;
# 	req_hit_way <= hit_way;
        # Generate the "hit" and "miss" signals for the synchronous blocks
        with m.If(i_in.rq & access_ok & ~flush_in & '''TODO nmigen rst'''):
            comb += req_is_hit.eq(is_hit)
            comb += req_is_miss.eq(~is_hit)

        with m.Else():
            comb += req_is_hit.eq(0)
            comb += req_is_miss.eq(0)

#       -- The way to replace on a miss
#       if r.state = CLR_TAG then
#           replace_way <= to_integer(unsigned(plru_victim(r.store_index)));
#       else
#           replace_way <= r.store_way;
#       end if;
        # The way to replace on a miss
        with m.If(r.state == State.CLR_TAG):
            comb += replace_way.eq(plru_victim[r.store_index])

        with m.Else():
            comb += replace_way.eq(r.store_way)

# 	-- Output instruction from current cache row
# 	--
# 	-- Note: This is a mild violation of our design principle of
#       -- having pipeline stages output from a clean latch. In this
#       -- case we output the result of a mux. The alternative would
#       -- be output an entire row which I prefer not to do just yet
#       -- as it would force fetch2 to know about some of the cache
#       -- geometry information.
#       i_out.insn <= read_insn_word(r.hit_nia, cache_out(r.hit_way));
# 	i_out.valid <= r.hit_valid;
# 	i_out.nia <= r.hit_nia;
# 	i_out.stop_mark <= r.hit_smark;
#       i_out.fetch_failed <= r.fetch_failed;
        # Output instruction from current cache row
        #
        # Note: This is a mild violation of our design principle of
        # having pipeline stages output from a clean latch. In this
        # case we output the result of a mux. The alternative would
        # be output an entire row which I prefer not to do just yet
        # as it would force fetch2 to know about some of the cache
        # geometry information.
        comb += i_out.insn.eq(
                 read_insn_word(r.hit_nia, cache_out[r.hit_way])
                )
        comb += i_out.valid.eq(r.hit_valid)
        comb += i_out.nia.eq(r.hit_nia)
        comb += i_out.stop_mark.eq(r.hit_smark)
        comb += i_out.fetch_failed.eq(r.fetch_failed)

# 	-- Stall fetch1 if we have a miss on cache or TLB
#       -- or a protection fault
# 	stall_out <= not (is_hit and access_ok);
        # Stall fetch1 if we have a miss on cache or TLB
        # or a protection fault
        comb += stall_out.eq(~(is_hit & access_ok))

# 	-- Wishbone requests output (from the cache miss reload machine)
# 	wishbone_out <= r.wb;
        # Wishbone requests output (from the cache miss reload machine)
        comb += wb_out.eq(r.wb)
#     end process;

#     -- Cache hit synchronous machine
#     icache_hit : process(clk)
    # Cache hit synchronous machine
    def icache_hit(self, m):
        sync = m.d.sync
#     begin
#         if rising_edge(clk) then
#             -- keep outputs to fetch2 unchanged on a stall
#             -- except that flush or reset sets valid to 0
#             -- If use_previous, keep the same data as last
#             -- cycle and use the second half
#             if stall_in = '1' or use_previous = '1' then
#                 if rst = '1' or flush_in = '1' then
#                     r.hit_valid <= '0';
#             end if;
        # keep outputs to fetch2 unchanged on a stall
        # except that flush or reset sets valid to 0
        # If use_previous, keep the same data as last
        # cycle and use the second half
        with m.If(stall_in | use_previous):
            with m.If('''TODO rst nmigen''' | flush_in):
                sync += r.hit_valid.eq(0)
#             else
#                 -- On a hit, latch the request for the next cycle,
#                 -- when the BRAM data will be available on the
#                 -- cache_out output of the corresponding way
#                 r.hit_valid <= req_is_hit;
#                 if req_is_hit = '1' then
#                     r.hit_way <= req_hit_way;
        with m.Else():
            # On a hit, latch the request for the next cycle,
            # when the BRAM data will be available on the
            # cache_out output of the corresponding way
            sync += r.hit_valid.eq(req_is_hit)

            with m.If(req_is_hit):
                sync += r.hit_way.eq(req_hit_way)

#                     report "cache hit nia:" & to_hstring(i_in.nia) &
#                         " IR:" & std_ulogic'image(i_in.virt_mode) &
#                         " SM:" & std_ulogic'image(i_in.stop_mark) &
#                         " idx:" & integer'image(req_index) &
#                         " tag:" & to_hstring(req_tag) &
#                         " way:" & integer'image(req_hit_way) &
#                         " RA:" & to_hstring(real_addr);
                print(f"cache hit nia:{i_in.nia}, IR:{i_in.virt_mode}, " \
                      f"SM:{i_in.stop_mark}, idx:{req_index}, " \
                      f"tag:{req_tag}, way:{req_hit_way}, RA:{real_addr}")
#                 end if;
# 	    end if;
#             if stall_in = '0' then
#                 -- Send stop marks and NIA down regardless of validity
#                 r.hit_smark <= i_in.stop_mark;
#                 r.hit_nia <= i_in.nia;
#             end if;
        with m.If(~stall_in):
            # Send stop marks and NIA down regardless of validity
            sync += r.hit_smark.eq(i_in.stop_mark)
            sync += r.hit_nia.eq(i_in.nia)
# 	end if;
#     end process;

#     -- Cache miss/reload synchronous machine
#     icache_miss : process(clk)
    # Cache miss/reload synchronous machine
    def icache_miss(self, m):
        comb = m.d.comb
        sync = m.d.sync

# 	variable tagset    : cache_tags_set_t;
# 	variable stbs_done : boolean;

        tagset    = Signal(TAG_RAM_WIDTH)
        stbs_done = Signal()

#     begin
#         if rising_edge(clk) then
# 	    -- On reset, clear all valid bits to force misses
#             if rst = '1' then
        # On reset, clear all valid bits to force misses
        with m.If('''TODO rst nmigen'''):
# 		for i in index_t loop
# 		    cache_valids(i) <= (others => '0');
# 		end loop;
            for i in Signal(NUM_LINES):
                sync += cache_valid_bits[i].eq(~1)

#                 r.state <= IDLE;
#                 r.wb.cyc <= '0';
#                 r.wb.stb <= '0';
            sync += r.state.eq(State.IDLE)
            sync += r.wb.cyc.eq(0)
            sync += r.wb.stb.eq(0)

# 		-- We only ever do reads on wishbone
# 		r.wb.dat <= (others => '0');
# 		r.wb.sel <= "11111111";
# 		r.wb.we  <= '0';
            # We only ever do reads on wishbone
            sync += r.wb.dat.eq(~1)
            sync += r.wb.sel.eq(Const(0b11111111, 8))
            sync += r.wb.we.eq(0)

# 		-- Not useful normally but helps avoiding tons of sim warnings
# 		r.wb.adr <= (others => '0');
            # Not useful normally but helps avoiding tons of sim warnings
            sync += r.wb.adr.eq(~1)

#             else
        with m.Else():
#                 -- Process cache invalidations
#                 if inval_in = '1' then
#                     for i in index_t loop
#                         cache_valids(i) <= (others => '0');
#                     end loop;
#                     r.store_valid <= '0';
#                 end if;
            # Process cache invalidations
            with m.If(inval_in):
                for i in range(NUM_LINES):
                    sync += cache_valid_bits[i].eq(~1)

                sync += r.store_valid.eq(0)

# 		-- Main state machine
# 		case r.state is
                # Main state machine
                with m.Switch(r.state):

# 		when IDLE =>
                    with m.Case(State.IDLE):
#                     -- Reset per-row valid flags, only used in WAIT_ACK
#                     for i in 0 to ROW_PER_LINE - 1 loop
#                         r.rows_valid(i) <= '0';
#                     end loop;
                        # Reset per-row valid flags, onlyy used in WAIT_ACK
                        for i in range(ROW_PER_LINE):
                            sync += r.rows_valid[i].eq(0)

# 		    -- We need to read a cache line
# 		    if req_is_miss = '1' then
# 			report "cache miss nia:" & to_hstring(i_in.nia) &
#                             " IR:" & std_ulogic'image(i_in.virt_mode) &
# 			    " SM:" & std_ulogic'image(i_in.stop_mark) &
# 			    " idx:" & integer'image(req_index) &
# 			    " way:" & integer'image(replace_way) &
# 			    " tag:" & to_hstring(req_tag) &
#                             " RA:" & to_hstring(real_addr);
                        # We need to read a cache line
                        with m.If(req_is_miss):
                            print(f"cache miss nia:{i_in.nia} " \
                                  f"IR:{i_in.virt_mode} " \
                                  f"SM:{i_in.stop_mark} idx:{req_index} " \
                                  f"way:{replace_way} tag:{req_tag} " \
                                  f"RA:{real_addr}")

# 			-- Keep track of our index and way for
#                       -- subsequent stores
# 			r.store_index <= req_index;
# 			r.store_row <= get_row(req_laddr);
#                       r.store_tag <= req_tag;
#                       r.store_valid <= '1';
#                       r.end_row_ix <=
#                        get_row_of_line(get_row(req_laddr)) - 1;
                            # Keep track of our index and way
                            # for subsequent stores
                            sync += r.store_index.eq(req_index)
                            sync += r.store_row.eq(get_row(req_laddr))
                            sync += r.store_tag.eq(req_tag)
                            sync += r.store_valid.eq(1)
                            sync += r.end_row_ix.eq(
                                     get_row_of_line(get_row(req_laddr)) - 1
                                    )

# 			-- Prep for first wishbone read. We calculate the
#                       -- address of the start of the cache line and
#                       -- start the WB cycle.
# 			r.wb.adr <= req_laddr(r.wb.adr'left downto 0);
# 			r.wb.cyc <= '1';
# 			r.wb.stb <= '1';
                            # Prep for first wishbone read. We calculate the
                            # address of the start of the cache line and
                            # start the WB cycle.
                            sync += r.wb.adr.eq(
                                     req_laddr[:r.wb.adr '''left?''']
                                    )

# 			-- Track that we had one request sent
# 			r.state <= CLR_TAG;
                            # Track that we had one request sent
                            sync += r.state.eq(State.CLR_TAG)
# 		    end if;

# 		when CLR_TAG | WAIT_ACK =>
                    with m.Case(State.CLR_TAG, State.WAIT_ACK):
#                     if r.state = CLR_TAG then
                        with m.If(r.state == State.CLR_TAG):
#                         -- Get victim way from plru
# 			r.store_way <= replace_way;
                            # Get victim way from plru
                            sync += r.store_way.eq(replace_way)
#
# 			-- Force misses on that way while reloading that line
# 			cache_valids(req_index)(replace_way) <= '0';
                            # Force misses on that way while
                            # realoading that line
                            sync += cache_valid_bits[
                                     req_index
                                    ][replace_way].eq(0)

# 			-- Store new tag in selected way
# 			for i in 0 to NUM_WAYS-1 loop
# 			    if i = replace_way then
# 				tagset := cache_tags(r.store_index);
# 				write_tag(i, tagset, r.store_tag);
# 				cache_tags(r.store_index) <= tagset;
# 			    end if;
# 			end loop;
                            for i in range(NUM_WAYS):
                                with m.If(i == replace_way):
                                    comb += tagset.eq(
                                             cache_tags[r.store_index]
                                            )
                                    sync += write_tag(i, tagset, r.store_tag)
                                    sync += cache_tags(r.store_index).eq(
                                             tagset
                                            )

#                         r.state <= WAIT_ACK;
                            sync += r.state.eq(State.WAIT_ACK)
#                     end if;

# 		    -- Requests are all sent if stb is 0
# 		    stbs_done := r.wb.stb = '0';
                        # Requests are all sent if stb is 0
                        comb += stbs_done.eq(r.wb.stb == 0)

# 		    -- If we are still sending requests, was one accepted ?
# 		    if wishbone_in.stall = '0' and not stbs_done then
                        # If we are still sending requests, was one accepted?
                        with m.If(~wb_in.stall & ~stbs_done):
# 			-- That was the last word ? We are done sending.
#                       -- Clear stb and set stbs_done so we can handle
#                       -- an eventual last ack on the same cycle.
# 			if is_last_row_addr(r.wb.adr, r.end_row_ix) then
# 			    r.wb.stb <= '0';
# 			    stbs_done := true;
# 			end if;
                            # That was the last word ? We are done sending.
                            # Clear stb and set stbs_done so we can handle
                            # an eventual last ack on the same cycle.
                            with m.If(is_last_row_addr(
                                      r.wb.adr, r.end_row_ix)):
                                sync += r.wb.stb.eq(0)
                                stbs_done.eq(1)

# 			-- Calculate the next row address
# 			r.wb.adr <= next_row_addr(r.wb.adr);
                            # Calculate the next row address
                            sync += r.wb.adr.eq(next_row_addr(r.wb.adr))
# 		    end if;

# 		    -- Incoming acks processing
# 		    if wishbone_in.ack = '1' then
                        # Incoming acks processing
                        with m.If(wb_in.ack):
#                         r.rows_valid(r.store_row mod ROW_PER_LINE) <= '1';
                            sync += r.rows_valid[
                                     r.store_row & ROW_PER_LINE
                                    ].eq(1)

# 			-- Check for completion
# 			if stbs_done and
#                        is_last_row(r.store_row, r.end_row_ix) then
                            # Check for completion
                            with m.If(stbs_done & is_last_row(
                                      r.store_row, r.end_row_ix)):
# 			    -- Complete wishbone cycle
# 			    r.wb.cyc <= '0';
                                # Complete wishbone cycle
                                sync += r.wb.cyc.eq(0)

# 			    -- Cache line is now valid
# 			    cache_valids(r.store_index)(replace_way) <=
#                            r.store_valid and not inval_in;
                                # Cache line is now valid
                                sync += cache_valid_bits[
                                         r.store_index
                                        ][relace_way].eq(
                                         r.store_valid & ~inval_in
                                        )

# 			    -- We are done
# 			    r.state <= IDLE;
                                # We are done
                                sync += r.state.eq(State.IDLE)
# 			end if;

# 			-- Increment store row counter
# 			r.store_row <= next_row(r.store_row);
                            # Increment store row counter
                            sync += store_row.eq(next_row(r.store_row))
# 		    end if;
# 		end case;
# 	    end if;
#
#             -- TLB miss and protection fault processing
#             if rst = '1' or flush_in = '1' or m_in.tlbld = '1' then
#                 r.fetch_failed <= '0';
#             elsif i_in.req = '1' and access_ok = '0' and stall_in = '0' then
#                 r.fetch_failed <= '1';
#             end if;
            # TLB miss and protection fault processing
            with m.If('''TODO nmigen rst''' | flush_in | m_in.tlbld):
                sync += r.fetch_failed.eq(0)

            with m.Elif(i_in.req & ~access_ok & ~stall_in):
                sync += r.fetch_failed.eq(1)
# 	end if;
#     end process;

#     icache_log: if LOG_LENGTH > 0 generate
    def icache_log(self, m, log_out):
        comb = m.d.comb
        sync = m.d.sync

#         -- Output data to logger
#         signal log_data    : std_ulogic_vector(53 downto 0);
#     begin
#         data_log: process(clk)
#             variable lway: way_t;
#             variable wstate: std_ulogic;
        # Output data to logger
        for i in range(LOG_LENGTH)
            # Output data to logger
            log_data = Signal(54)
            lway     = Signal(NUM_WAYS)
            wstate   = Signal()

#         begin
#             if rising_edge(clk) then
#                 lway := req_hit_way;
#                 wstate := '0';
            comb += lway.eq(req_hit_way)
            comb += wstate.eq(0)

#                 if r.state /= IDLE then
#                     wstate := '1';
#                 end if;
            with m.If(r.state != State.IDLE):
                comb += wstate.eq(1)

#                 log_data <= i_out.valid &
#                             i_out.insn &
#                             wishbone_in.ack &
#                             r.wb.adr(5 downto 3) &
#                             r.wb.stb & r.wb.cyc &
#                             wishbone_in.stall &
#                             stall_out &
#                             r.fetch_failed &
#                             r.hit_nia(5 downto 2) &
#                             wstate &
#                             std_ulogic_vector(to_unsigned(lway, 3)) &
#                             req_is_hit & req_is_miss &
#                             access_ok &
#                             ra_valid;
            sync += log_data.eq(Cat(
                     ra_valid, access_ok, req_is_miss, req_is_hit,
                     lway '''truncate to 3 bits?''', wstate, r.hit_nia[2:6],
                     r.fetch_failed, stall_out, wb_in.stall, r.wb.cyc,
                     r.wb.stb, r.wb.adr[3:6], wb_in.ack, i_out.insn,
                     i_out.valid
                    ))
#             end if;
#         end process;
#         log_out <= log_data;
            comb += log_out.eq(log_data)
#     end generate;
# end;

    def elaborate(self, platform):
# architecture rtl of icache is
#     constant ROW_SIZE_BITS : natural := ROW_SIZE*8;
#     -- ROW_PER_LINE is the number of row (wishbone transactions) in a line
#     constant ROW_PER_LINE  : natural := LINE_SIZE / ROW_SIZE;
#     -- BRAM_ROWS is the number of rows in BRAM needed to represent the full
#     -- icache
#     constant BRAM_ROWS     : natural := NUM_LINES * ROW_PER_LINE;
#     -- INSN_PER_ROW is the number of 32bit instructions per BRAM row
#     constant INSN_PER_ROW  : natural := ROW_SIZE_BITS / 32;
#     -- Bit fields counts in the address
#
#     -- INSN_BITS is the number of bits to select an instruction in a row
#     constant INSN_BITS     : natural := log2(INSN_PER_ROW);
#     -- ROW_BITS is the number of bits to select a row
#     constant ROW_BITS      : natural := log2(BRAM_ROWS);
#     -- ROW_LINEBITS is the number of bits to select a row within a line
#     constant ROW_LINEBITS  : natural := log2(ROW_PER_LINE);
#     -- LINE_OFF_BITS is the number of bits for the offset in a cache line
#     constant LINE_OFF_BITS : natural := log2(LINE_SIZE);
#     -- ROW_OFF_BITS is the number of bits for the offset in a row
#     constant ROW_OFF_BITS  : natural := log2(ROW_SIZE);
#     -- INDEX_BITS is the number of bits to select a cache line
#     constant INDEX_BITS    : natural := log2(NUM_LINES);
#     -- SET_SIZE_BITS is the log base 2 of the set size
#     constant SET_SIZE_BITS : natural := LINE_OFF_BITS + INDEX_BITS;
#     -- TAG_BITS is the number of bits of the tag part of the address
#     constant TAG_BITS      : natural := REAL_ADDR_BITS - SET_SIZE_BITS;
#     -- WAY_BITS is the number of bits to select a way
#     constant WAY_BITS     : natural := log2(NUM_WAYS);

        ROW_SIZE_BITS  = ROW_SIZE * 8
        # ROW_PER_LINE is the number of row
        # (wishbone) transactions in a line
        ROW_PER_LINE   = LINE_SIZE / ROW_SIZE
        # BRAM_ROWS is the number of rows in
        # BRAM needed to represent the full icache
        BRAM_ROWS      = NUM_LINES * ROW_PER_LINE
        # INSN_PER_ROW is the number of 32bit
        # instructions per BRAM row
        INSN_PER_ROW   = ROW_SIZE_BITS / 32

        # Bit fields counts in the address
        #
        # INSN_BITS is the number of bits to
        # select an instruction in a row
        INSN_BITS      = log2_int(INSN_PER_ROW)
        # ROW_BITS is the number of bits to
        # select a row
        ROW_BITS       = log2_int(BRAM_ROWS)
        # ROW_LINEBITS is the number of bits to
        # select a row within a line
        ROW_LINE_BITS   = log2_int(ROW_PER_LINE)
        # LINE_OFF_BITS is the number of bits for
        # the offset in a cache line
        LINE_OFF_BITS  = log2_int(LINE_SIZE)
        # ROW_OFF_BITS is the number of bits for
        # the offset in a row
        ROW_OFF_BITS   = log2_int(ROW_SIZE)
        # INDEX_BITS is the number of bits to
        # select a cache line
        INDEX_BITS     = log2_int(NUM_LINES)
        # SET_SIZE_BITS is the log base 2 of
        # the set size
        SET_SIZE_BITS  = LINE_OFF_BITS + INDEX_BITS
        # TAG_BITS is the number of bits of
        # the tag part of the address
        TAG_BITS       = REAL_ADDR_BITS - SET_SIZE_BITS
        # WAY_BITS is the number of bits to
        # select a way
        WAY_BITS       = log2_int(NUM_WAYS)
        TAG_RAM_WIDTH  = TAG_BITS * NUM_WAYS

#     -- Example of layout for 32 lines of 64 bytes:
#     --
#     -- ..  tag    |index|  line  |
#     -- ..         |   row   |    |
#     -- ..         |     |   | |00| zero          (2)
#     -- ..         |     |   |-|  | INSN_BITS     (1)
#     -- ..         |     |---|    | ROW_LINEBITS  (3)
#     -- ..         |     |--- - --| LINE_OFF_BITS (6)
#     -- ..         |         |- --| ROW_OFF_BITS  (3)
#     -- ..         |----- ---|    | ROW_BITS      (8)
#     -- ..         |-----|        | INDEX_BITS    (5)
#     -- .. --------|              | TAG_BITS      (53)
        # Example of layout for 32 lines of 64 bytes:
        #
        # ..  tag    |index|  line  |
        # ..         |   row   |    |
        # ..         |     |   | |00| zero          (2)
        # ..         |     |   |-|  | INSN_BITS     (1)
        # ..         |     |---|    | ROW_LINEBITS  (3)
        # ..         |     |--- - --| LINE_OFF_BITS (6)
        # ..         |         |- --| ROW_OFF_BITS  (3)
        # ..         |----- ---|    | ROW_BITS      (8)
        # ..         |-----|        | INDEX_BITS    (5)
        # .. --------|              | TAG_BITS      (53)

#     subtype row_t is integer range 0 to BRAM_ROWS-1;
#     subtype index_t is integer range 0 to NUM_LINES-1;
#     subtype way_t is integer range 0 to NUM_WAYS-1;
#     subtype row_in_line_t is unsigned(ROW_LINEBITS-1 downto 0);
#
#     -- The cache data BRAM organized as described above for each way
#     subtype cache_row_t is std_ulogic_vector(ROW_SIZE_BITS-1 downto 0);
#
#     -- The cache tags LUTRAM has a row per set. Vivado is a pain and will
#     -- not handle a clean (commented) definition of the cache tags as a 3d
#     -- memory. For now, work around it by putting all the tags
#     subtype cache_tag_t is std_logic_vector(TAG_BITS-1 downto 0);
# --    type cache_tags_set_t is array(way_t) of cache_tag_t;
# --    type cache_tags_array_t is array(index_t) of cache_tags_set_t;
#     constant TAG_RAM_WIDTH : natural := TAG_BITS * NUM_WAYS;
#     subtype cache_tags_set_t is std_logic_vector(TAG_RAM_WIDTH-1 downto 0);
#     type cache_tags_array_t is array(index_t) of cache_tags_set_t;
        def CacheTagArray():
            return Array(Signal(TAG_RAM_WIDTH) for x in range(NUM_LINES))

#     -- The cache valid bits
#     subtype cache_way_valids_t is std_ulogic_vector(NUM_WAYS-1 downto 0);
#     type cache_valids_t is array(index_t) of cache_way_valids_t;
#     type row_per_line_valid_t is array(0 to ROW_PER_LINE - 1) of std_ulogic;
        def CacheValidBitsArray():
            return Array(Signal() for x in ROW_PER_LINE)

        def RowPerLineValidArray():
            return Array(Signal() for x in range ROW_PER_LINE)

#     -- Storage. Hopefully "cache_rows" is a BRAM, the rest is LUTs
#     signal cache_tags   : cache_tags_array_t;
#     signal cache_valids : cache_valids_t;
        # Storage. Hopefully "cache_rows" is a BRAM, the rest is LUTs
        cache_tags = CacheTagArray()
        cache_valid_bits = CacheValidBitsArray()

#     attribute ram_style : string;
#     attribute ram_style of cache_tags : signal is "distributed";
        # TODO to be passed to nigmen as ram attributes
        # attribute ram_style : string;
        # attribute ram_style of cache_tags : signal is "distributed";

#     -- L1 ITLB.
#     constant TLB_BITS : natural := log2(TLB_SIZE);
#     constant TLB_EA_TAG_BITS : natural := 64 - (TLB_LG_PGSZ + TLB_BITS);
#     constant TLB_PTE_BITS : natural := 64;
        TLB_BITS        = log2_int(TLB_SIZE)
        TLB_EA_TAG_BITS = 64 - (TLB_LG_PGSZ + TLB_BITS)
        TLB_PTE_BITS    = 64

#     subtype tlb_index_t is integer range 0 to TLB_SIZE - 1;
#     type tlb_valids_t is array(tlb_index_t) of std_ulogic;
#     subtype tlb_tag_t is std_ulogic_vector(TLB_EA_TAG_BITS - 1 downto 0);
#     type tlb_tags_t is array(tlb_index_t) of tlb_tag_t;
#     subtype tlb_pte_t is std_ulogic_vector(TLB_PTE_BITS - 1 downto 0);
#     type tlb_ptes_t is array(tlb_index_t) of tlb_pte_t;
        def TLBValidBitsArray():
            return Array(Signal() for x in range(TLB_SIZE))

        def TLBTagArray():
            return Array(Signal(TLB_EA_TAG_BITS) for x in range(TLB_SIZE))

        def TLBPTEArray():
            return Array(Signal(LTB_PTE_BITS) for x in range(TLB_SIZE))

#     signal itlb_valids : tlb_valids_t;
#     signal itlb_tags : tlb_tags_t;
#     signal itlb_ptes : tlb_ptes_t;
#     attribute ram_style of itlb_tags : signal is "distributed";
#     attribute ram_style of itlb_ptes : signal is "distributed";
        itlb_valid_bits = TLBValidBitsArray()
        itlb_tags       = TLBTagArray()
        itlb_ptes       = TLBPTEArray()
        # TODO to be passed to nmigen as ram attributes
        # attribute ram_style of itlb_tags : signal is "distributed";
        # attribute ram_style of itlb_ptes : signal is "distributed";

#     -- Privilege bit from PTE EAA field
#     signal eaa_priv  : std_ulogic;
        # Privilege bit from PTE EAA field
        eaa_priv        = Signal()


#     signal r : reg_internal_t;
        r = RegInternal()

#     -- Async signals on incoming request
#     signal req_index   : index_t;
#     signal req_row     : row_t;
#     signal req_hit_way : way_t;
#     signal req_tag     : cache_tag_t;
#     signal req_is_hit  : std_ulogic;
#     signal req_is_miss : std_ulogic;
#     signal req_laddr   : std_ulogic_vector(63 downto 0);
        # Async signal on incoming request
        req_index     = Signal(NUM_LINES)
        req_row       = Signal(BRAM_ROWS)
        req_hit_way   = Signal(NUM_WAYS)
        req_tag       = Signal(TAG_BITS)
        req_is_hit    = Signal()
        req_is_miss   = Signal()
        req_laddr     = Signal(64)

#     signal tlb_req_index : tlb_index_t;
#     signal real_addr     : std_ulogic_vector(REAL_ADDR_BITS - 1 downto 0);
#     signal ra_valid      : std_ulogic;
#     signal priv_fault    : std_ulogic;
#     signal access_ok     : std_ulogic;
#     signal use_previous  : std_ulogic;
        tlb_req_index = Signal(TLB_SIZE)
        real_addr     = Signal(REAL_ADDR_BITS)
        ra_valid      = Signal()
        priv_fault    = Signal()
        access_ok     = Signal()
        use_previous  = Signal()

#     -- Cache RAM interface
#     type cache_ram_out_t is array(way_t) of cache_row_t;
#     signal cache_out   : cache_ram_out_t;
        # Cache RAM interface
        def CacheRamOut():
            return Array(Signal(ROW_SIZE_BITS) for x in range(NUM_WAYS))

        cache_out     = CacheRamOut()

#     -- PLRU output interface
#     type plru_out_t is array(index_t) of
#      std_ulogic_vector(WAY_BITS-1 downto 0);
#     signal plru_victim : plru_out_t;
#     signal replace_way : way_t;
        # PLRU output interface
        def PLRUOut():
            return Array(Signal(WAY_BITS) for x in range(NUM_LINES))

        plru_victim   = PLRUOut()
        replace_way   = Signal(NUM_WAYS)

# begin
#
#     assert LINE_SIZE mod ROW_SIZE = 0;
#     assert ispow2(LINE_SIZE) report "LINE_SIZE not power of 2"
#      severity FAILURE;
#     assert ispow2(NUM_LINES) report "NUM_LINES not power of 2"
#      severity FAILURE;
#     assert ispow2(ROW_PER_LINE) report "ROW_PER_LINE not power of 2"
#      severity FAILURE;
#     assert ispow2(INSN_PER_ROW) report "INSN_PER_ROW not power of 2"
#      severity FAILURE;
#     assert (ROW_BITS = INDEX_BITS + ROW_LINEBITS)
# 	report "geometry bits don't add up" severity FAILURE;
#     assert (LINE_OFF_BITS = ROW_OFF_BITS + ROW_LINEBITS)
# 	report "geometry bits don't add up" severity FAILURE;
#     assert (REAL_ADDR_BITS = TAG_BITS + INDEX_BITS + LINE_OFF_BITS)
# 	report "geometry bits don't add up" severity FAILURE;
#     assert (REAL_ADDR_BITS = TAG_BITS + ROW_BITS + ROW_OFF_BITS)
# 	report "geometry bits don't add up" severity FAILURE;
#
#     sim_debug: if SIM generate
#     debug: process
#     begin
# 	report "ROW_SIZE      = " & natural'image(ROW_SIZE);
# 	report "ROW_PER_LINE  = " & natural'image(ROW_PER_LINE);
# 	report "BRAM_ROWS     = " & natural'image(BRAM_ROWS);
# 	report "INSN_PER_ROW  = " & natural'image(INSN_PER_ROW);
# 	report "INSN_BITS     = " & natural'image(INSN_BITS);
# 	report "ROW_BITS      = " & natural'image(ROW_BITS);
# 	report "ROW_LINEBITS  = " & natural'image(ROW_LINEBITS);
# 	report "LINE_OFF_BITS = " & natural'image(LINE_OFF_BITS);
# 	report "ROW_OFF_BITS  = " & natural'image(ROW_OFF_BITS);
# 	report "INDEX_BITS    = " & natural'image(INDEX_BITS);
# 	report "TAG_BITS      = " & natural'image(TAG_BITS);
# 	report "WAY_BITS      = " & natural'image(WAY_BITS);
# 	wait;
#     end process;
#     end generate;

