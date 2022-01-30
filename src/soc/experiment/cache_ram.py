# TODO: replace with Memory at some point
from nmigen import Elaboratable, Signal, Array, Module, Memory
from nmutil.util import Display


class CacheRam(Elaboratable):

    def __init__(self, ROW_BITS=16, WIDTH = 64, TRACE=True, ADD_BUF=False,
                       ram_num=0):
        self.ram_num = ram_num # for debug reporting
        self.ROW_BITS = ROW_BITS
        self.WIDTH = WIDTH
        self.TRACE = TRACE
        self.ADD_BUF = ADD_BUF
        self.rd_en     = Signal()
        self.rd_addr   = Signal(ROW_BITS)
        self.rd_data_o = Signal(WIDTH)
        self.wr_sel    = Signal(WIDTH//8)
        self.wr_addr   = Signal(ROW_BITS)
        self.wr_data   = Signal(WIDTH)
 
    def elaborate(self, platform):
        m = Module()
        comb, sync = m.d.comb, m.d.sync

        ROW_BITS = self.ROW_BITS
        WIDTH = self.WIDTH
        TRACE = self.TRACE
        ADD_BUF = self.ADD_BUF
        SIZE = 2**ROW_BITS
     
        # set up the Cache RAM Memory and create one read and one write port
        # the read port is *not* transparent (does not pass write-thru-read)
        #attribute ram_style of ram : signal is "block";
        ram = Memory(depth=SIZE, width=WIDTH)
        m.submodules.rdport = rdport = ram.read_port(transparent=False)
        m.submodules.wrport = wrport = ram.write_port(granularity=8)

        with m.If(TRACE):
            with m.If(self.wr_sel.bool()):
                sync += Display( "write ramno %d a: %%x "
                                 "sel: %%x dat: %%x" % self.ram_num,
                                self.wr_addr,
                                self.wr_sel, self.wr_data)

        # read data output and a latched copy. behaves like microwatt cacheram
        rd_data0 = Signal(WIDTH)
        rd_data0l = Signal(WIDTH)

        # delay on read address/en
        rd_delay = Signal()
        rd_delay_addr = Signal.like(self.rd_addr)
        sync += rd_delay_addr.eq(self.rd_addr)
        sync += rd_delay.eq(self.rd_en)

        # write port
        comb += wrport.addr.eq(self.wr_addr)
        comb += wrport.en.eq(self.wr_sel)
        comb += wrport.data.eq(self.wr_data)

        # read port (include a latch on the output, for microwatt compatibility)
        comb += rdport.addr.eq(self.rd_addr)
        comb += rdport.en.eq(self.rd_en)
        with m.If(rd_delay):
            comb += rd_data0.eq(rdport.data)
            sync += rd_data0l.eq(rd_data0)   # preserve latched data
        with m.Else():
            comb += rd_data0.eq(rd_data0l)   # output latched (last-read)

        if TRACE:
            with m.If(rd_delay):
                sync += Display("read ramno %d a: %%x dat: %%x" % self.ram_num,
                                rd_delay_addr, rd_data0)
                pass

        # extra delay requested?
        if ADD_BUF:
            sync += self.rd_data_o.eq(rd_data0)
        else:
            comb += self.rd_data_o.eq(rd_data0)

        return m
