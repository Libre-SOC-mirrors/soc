"""L0 Cache/Buffer

This first version is intended for prototyping and test purposes:
it has "direct" access to Memory.

The intention is that this version remains an integral part of the
test infrastructure, and, just as with minerva's memory arrangement,
a dynamic runtime config *selects* alternative memory arrangements
rather than *replaces and discards* this code.

Links:

* https://bugs.libre-soc.org/show_bug.cgi?id=216
* https://libre-soc.org/3d_gpu/architecture/memory_and_cache/

"""

from nmigen.compat.sim import run_simulation, Settle
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Mux, Elaboratable, Array, Cat
from nmutil.iocontrol import RecordObject
from nmigen.utils import log2_int
from nmigen.hdl.rec import Record, Layout

from nmutil.latch import SRLatch, latchregister
from soc.decoder.power_decoder2 import Data
from soc.decoder.power_enums import InternalOp
from soc.regfile.regfile import ortreereduce
from nmutil.util import treereduce

from soc.decoder.power_decoder2 import Data
#from nmutil.picker import PriorityPicker
from nmigen.lib.coding import PriorityEncoder
from soc.scoreboard.addr_split import LDSTSplitter
from soc.scoreboard.addr_match import LenExpand

# for testing purposes
from soc.config.test.test_loadstore import TestMemPspec
from soc.config.loadstore import ConfigMemoryPortInterface
from soc.experiment.pimem import PortInterface
from soc.config.test.test_pi2ls import pi_ld, pi_st, pi_ldst
import unittest


class DualPortSplitter(Elaboratable):
    """DualPortSplitter

    * one incoming PortInterface
    * two *OUTGOING* PortInterfaces
    * uses LDSTSplitter to do it

    (actually, thinking about it LDSTSplitter could simply be
     modified to conform to PortInterface: one in, two out)

    once that is done each pair of ports may be wired directly
    to the dual ports of L0CacheBuffer

    The split is carried out so that, regardless of alignment or
    mis-alignment, outgoing PortInterface[0] takes bit 4 == 0
    of the address, whilst outgoing PortInterface[1] takes
    bit 4 == 1.

    PortInterface *may* need to be changed so that the length is
    a binary number (accepting values 1-16).
    """
    def __init__(self):
        self.outp = [PortInterface(name="outp_0"),
                     PortInterface(name="outp_1")]
        self.inp  = PortInterface(name="inp")
        print(self.outp)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        m.submodules.splitter = splitter = LDSTSplitter(64, 48, 4)
        comb += splitter.addr_i.eq(self.inp.addr) #XXX
        #comb += splitter.len_i.eq()
        #comb += splitter.valid_i.eq()
        comb += splitter.is_ld_i.eq(self.inp.is_ld_i)
        comb += splitter.is_st_i.eq(self.inp.is_st_i)
        #comb += splitter.st_data_i.eq()
        #comb += splitter.sld_valid_i.eq()
        #comb += splitter.sld_data_i.eq()
        #comb += splitter.sst_valid_i.eq()
        return m


class DataMergerRecord(Record):
    """
    {data: 128 bit, byte_enable: 16 bit}
    """

    def __init__(self, name=None):
        layout = (('data', 128),
                  ('en', 16))
        Record.__init__(self, Layout(layout), name=name)

        self.data.reset_less = True
        self.en.reset_less = True


# TODO: formal verification
class DataMerger(Elaboratable):
    """DataMerger

    Merges data based on an address-match matrix.
    Identifies (picks) one (any) row, then uses that row,
    based on matching address bits, to merge (OR) all data
    rows into the output.

    Basically, by the time DataMerger is used, all of its incoming data is
    determined not to conflict.  The last step before actually submitting
    the request to the Memory Subsystem is to work out which requests,
    on the same 128-bit cache line, can be "merged" due to them being:
    (A) on the same address (bits 4 and above) (B) having byte-enable
    lines that (as previously mentioned) do not conflict.

    Therefore, put simply, this module will:
    (1) pick a row (any row) and identify it by an index labelled "idx"
    (2) merge all byte-enable lines which are on that same address, as
        indicated by addr_match_i[idx], onto the output
    """

    def __init__(self, array_size):
        """
        :addr_array_i: an NxN Array of Signals with bits set indicating address
                       match.  bits across the diagonal (addr_array_i[x][x])
                       will always be set, to indicate "active".
        :data_i: an Nx Array of Records {data: 128 bit, byte_enable: 16 bit}
        :data_o: an Output Record of same type
                 {data: 128 bit, byte_enable: 16 bit}
        """
        self.array_size = array_size
        ul = []
        for i in range(array_size):
            ul.append(Signal(array_size,
                             reset_less=True,
                             name="addr_match_%d" % i))
        self.addr_array_i = Array(ul)

        ul = []
        for i in range(array_size):
            ul.append(DataMergerRecord())
        self.data_i = Array(ul)
        self.data_o = DataMergerRecord()

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        #(1) pick a row
        m.submodules.pick = pick = PriorityEncoder(self.array_size)
        for j in range(self.array_size):
            comb += pick.i[j].eq(self.addr_array_i[j].bool())
        valid = ~pick.n
        idx = pick.o
        #(2) merge
        with m.If(valid):
            l = []
            for j in range(self.array_size):
                select = self.addr_array_i[idx][j]
                r = DataMergerRecord()
                with m.If(select):
                    comb += r.eq(self.data_i[j])
                l.append(r)
            comb += self.data_o.data.eq(ortreereduce(l,"data"))
            comb += self.data_o.en.eq(ortreereduce(l,"en"))

        return m


class L0CacheBuffer(Elaboratable):
    """L0 Cache / Buffer

    Note that the final version will have *two* interfaces per LDSTCompUnit,
    to cover mis-aligned requests, as well as *two* 128-bit L1 Cache
    interfaces: one for odd (addr[4] == 1) and one for even (addr[4] == 1).

    This version is to be used for test purposes (and actively maintained
    for such, rather than "replaced")

    There are much better ways to implement this.  However it's only
    a "demo" / "test" class, and one important aspect: it responds
    combinatorially, where a nmigen FSM's state-changes only activate
    on clock-sync boundaries.

    Note: the data byte-order is *not* expected to be normalised (LE/BE)
    by this class.  That task is taken care of by LDSTCompUnit.
    """

    def __init__(self, n_units, pimem, regwid=64, addrwid=48):
        self.n_units = n_units
        self.pimem = pimem
        self.regwid = regwid
        self.addrwid = addrwid
        ul = []
        for i in range(n_units):
            ul.append(PortInterface("ldst_port%d" % i, regwid, addrwid))
        self.dports = Array(ul)

    def elaborate(self, platform):
        m = Module()
        comb, sync = m.d.comb, m.d.sync

        # connect the ports as modules
        #for i in range(self.n_units):
        #    setattr(m.submodules, "port%d" % i, self.dports[i])

        # state-machine latches
        m.submodules.idx_l = idx_l = SRLatch(False, name="idx_l")
        m.submodules.reset_l = reset_l = SRLatch(True, name="reset")

        # find one LD (or ST) and do it.  only one per cycle.
        # TODO: in the "live" (production) L0Cache/Buffer, merge multiple
        # LD/STs using mask-expansion - see LenExpand class

        m.submodules.pick = pick = PriorityEncoder(self.n_units)

        ldsti = []
        for i in range(self.n_units):
            pi = self.dports[i]
            busy = (pi.is_ld_i | pi.is_st_i)# & pi.busy_o
            ldsti.append(busy) # accumulate ld/st-req
        # put the requests into the priority-picker
        comb += pick.i.eq(Cat(*ldsti))

        # hmm, have to select (record) the right port index
        nbits = log2_int(self.n_units, False)
        idx = Signal(nbits, reset_less=False)

        # use these because of the sync-and-comb pass-through capability
        latchregister(m, pick.o, idx, idx_l.q, name="idx_l")

        # convenience variables to reference the "picked" port
        port = self.dports[idx]

        # pick (and capture) the port index
        with m.If(~pick.n):
            comb += idx_l.s.eq(1)

        # from this point onwards, with the port "picked", it stays picked
        # until idx_l is deasserted
        comb += reset_l.s.eq(0)
        comb += reset_l.r.eq(0)

        with m.If(idx_l.q):
            comb += self.pimem.connect_port(port)
            with m.If(~self.pimem.pi.busy_o):
                comb += reset_l.s.eq(1) # reset when no longer busy

        # ugly hack, due to simultaneous addr req-go acknowledge
        reset_delay = Signal(reset_less=True)
        sync += reset_delay.eq(reset_l.q)

        # after waiting one cycle (reset_l is "sync" mode), reset the port
        with m.If(reset_l.q):
            comb += idx_l.r.eq(1)  # deactivate port-index selector
            comb += reset_l.r.eq(1)     # clear reset

        return m

    def ports(self):
        for p in self.dports:
            yield from p.ports()


class TstL0CacheBuffer(Elaboratable):
    def __init__(self, pspec, n_units=3):
        regwid = pspec.reg_wid
        addrwid = pspec.addr_wid
        self.cmpi = ConfigMemoryPortInterface(pspec)
        self.pimem = self.cmpi.pi
        self.l0 = L0CacheBuffer(n_units, self.pimem, regwid, addrwid<<1)

    def elaborate(self, platform):
        m = Module()
        m.submodules.pimem = self.pimem
        m.submodules.l0 = self.l0
        if hasattr(self.cmpi, 'lsmem'): # hmmm not happy about this
            m.submodules.lsmem = self.cmpi.lsmem.lsi

        return m

    def ports(self):
        yield from self.cmpi.ports()
        yield from self.l0.ports()
        yield from self.pimem.ports()


def wait_busy(port, no=False):
    while True:
        busy = yield port.busy_o
        print("busy", no, busy)
        if bool(busy) == no:
            break
        yield


def wait_addr(port):
    while True:
        addr_ok = yield port.addr_ok_o
        print("addrok", addr_ok)
        if not addr_ok:
            break
        yield


def wait_ldok(port):
    while True:
        ldok = yield port.ld.ok
        print("ldok", ldok)
        if ldok:
            break
        yield


def l0_cache_st(dut, addr, data, datalen):
    return pi_st(dut.l0, addr, datalen)


def l0_cache_ld(dut, addr, datalen, expected):
    return pi_ld(dut.l0, addr, datalen)


def l0_cache_ldst(arg, dut):
    port0 = dut.l0.dports[0]
    return pi_ldst(arg, port0)


def data_merger_merge(dut):
    print("data_merger")
    #starting with all inputs zero
    yield Settle()
    en = yield dut.data_o.en
    data = yield dut.data_o.data
    assert en == 0, "en must be zero"
    assert data == 0, "data must be zero"
    yield

    yield dut.addr_array_i[0].eq(0xFF)
    for j in range(dut.array_size):
        yield dut.data_i[j].en.eq(1 << j)
        yield dut.data_i[j].data.eq(0xFF << (16*j))
    yield Settle()

    en = yield dut.data_o.en
    data = yield dut.data_o.data
    assert data == 0xff00ff00ff00ff00ff00ff00ff00ff
    assert en == 0xff
    yield


class TestL0Cache(unittest.TestCase):

    def test_l0_cache_test_bare_wb(self):

        pspec = TestMemPspec(ldst_ifacetype='test_bare_wb',
                             addr_wid=48,
                             mask_wid=8,
                             reg_wid=64)
        dut = TstL0CacheBuffer(pspec)
        vl = rtlil.convert(dut, ports=[])# TODOdut.ports())
        with open("test_basic_l0_cache_bare_wb.il", "w") as f:
            f.write(vl)

        run_simulation(dut, l0_cache_ldst(self, dut),
                       vcd_name='test_l0_cache_basic_bare_wb.vcd')

    def test_l0_cache_testpi(self):

        pspec = TestMemPspec(ldst_ifacetype='testpi',
                             addr_wid=48,
                             mask_wid=8,
                             reg_wid=64)
        dut = TstL0CacheBuffer(pspec)
        vl = rtlil.convert(dut, ports=[])# TODOdut.ports())
        with open("test_basic_l0_cache.il", "w") as f:
            f.write(vl)

        run_simulation(dut, l0_cache_ldst(self, dut),
                       vcd_name='test_l0_cache_basic_testpi.vcd')


class TestDataMerger(unittest.TestCase):

    def test_data_merger(self):

        dut = DataMerger(8)
        #vl = rtlil.convert(dut, ports=dut.ports())
        #with open("test_data_merger.il", "w") as f:
        #    f.write(vl)

        run_simulation(dut, data_merger_merge(dut),
                       vcd_name='test_data_merger.vcd')


class TestDualPortSplitter(unittest.TestCase):

    def test_dual_port_splitter(self):

        dut = DualPortSplitter()
        #vl = rtlil.convert(dut, ports=dut.ports())
        #with open("test_data_merger.il", "w") as f:
        #    f.write(vl)

        #run_simulation(dut, data_merger_merge(dut),
        #               vcd_name='test_dual_port_splitter.vcd')


if __name__ == '__main__':
    unittest.main(exit=False)

