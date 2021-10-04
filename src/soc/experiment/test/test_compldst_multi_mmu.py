# test case for LOAD / STORE Computation Unit using MMU

#from nmigen.compat.sim import run_simulation
from nmigen.sim import Simulator, Delay, Settle
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Mux, Cat, Elaboratable, Array, Repl
from nmigen.hdl.rec import Record, Layout

from nmutil.latch import SRLatch, latchregister
from nmutil.byterev import byte_reverse
from nmutil.extend import exts
from nmutil.util import wrap
from soc.fu.regspec import RegSpecAPI

from openpower.decoder.power_enums import MicrOp, Function, LDSTMode
from soc.fu.ldst.ldst_input_record import CompLDSTOpSubset
from openpower.decoder.power_decoder2 import Data
from openpower.consts import MSR

from soc.experiment.compalu_multi import go_record, CompUnitRecord
from soc.experiment.l0_cache import PortInterface
from soc.experiment.pimem import LDSTException
from soc.experiment.compldst_multi import LDSTCompUnit, load, store
from soc.config.test.test_loadstore import TestMemPspec

from soc.experiment.mmu import MMU
from nmutil.util import Display

from soc.config.loadstore import ConfigMemoryPortInterface
from soc.experiment.test import pagetables


def wait_for_debug(sig, event, wait=True, test1st=False):
    v = (yield sig)
    print("wait for", sig, v, wait, test1st)
    if test1st and bool(v) == wait:
        return
    while True:
        yield
        v = (yield sig)
        yield Display("waiting for "+event)
        if bool(v) == wait:
            break

def load_debug(dut, src1, src2, imm, imm_ok=True, update=False, zero_a=False,
         byterev=True):
    print("LD", src1, src2, imm, imm_ok, update)
    yield dut.oper_i.insn_type.eq(MicrOp.OP_LOAD)
    yield dut.oper_i.data_len.eq(2)  # half-word
    yield dut.oper_i.byte_reverse.eq(byterev)
    yield dut.src1_i.eq(src1)
    yield dut.src2_i.eq(src2)
    yield dut.oper_i.zero_a.eq(zero_a)
    yield dut.oper_i.imm_data.data.eq(imm)
    yield dut.oper_i.imm_data.ok.eq(imm_ok)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield

    # set up read-operand flags
    rd = 0b00
    if not imm_ok:  # no immediate means RB register needs to be read
        rd |= 0b10
    if not zero_a:  # no zero-a means RA needs to be read
        rd |= 0b01

    # wait for the operands (RA, RB, or both)
    if rd:
        yield dut.rd.go_i.eq(rd)
        yield from wait_for_debug(dut.rd.rel_o,"operands")
        yield dut.rd.go_i.eq(0)

    yield from wait_for_debug(dut.adr_rel_o, "adr_rel_o" ,False, test1st=True)
    yield Display("load_debug: done")
    # yield dut.ad.go.eq(1)
    # yield
    # yield dut.ad.go.eq(0)

    """
    guess: hangs here

    if update:
        yield from wait_for(dut.wr.rel_o[1])
        yield dut.wr.go.eq(0b10)
        yield
        addr = yield dut.addr_o
        print("addr", addr)
        yield dut.wr.go.eq(0)
    else:
        addr = None

    yield from wait_for(dut.wr.rel_o[0], test1st=True)
    yield dut.wr.go.eq(1)
    yield
    data = yield dut.o_data
    print(data)
    yield dut.wr.go.eq(0)
    yield from wait_for(dut.busy_o)
    yield
    # wait_for(dut.stwd_mem_o)
    return data, addr
    """

# removed

# same thing as soc/src/soc/experiment/test/test_dcbz_pi.py
def ldst_sim(dut):
    yield dut.mmu.rin.prtbl.eq(0x1000000) # set process table
    ###yield from dcbz(dut, 4, 0, 3) # EA=7
    addr = 0x100e0
    data = 0xf553b658ba7e1f51

    yield from store(dut, addr, 0, data, 0)
    yield
    yield from load_debug(dut, 4, 0, 2) #FIXME
    """
    ld_data = yield from pi_ld(pi, addr, 8, msr_pr=0)
    assert ld_data == 0xf553b658ba7e1f51
    ld_data = yield from pi_ld(pi, addr, 8, msr_pr=0)
    assert ld_data == 0xf553b658ba7e1f51
    """
    yield

########################################


class TestLDSTCompUnitMMU(LDSTCompUnit):

    def __init__(self, rwid, pspec):
        from soc.experiment.l0_cache import TstL0CacheBuffer
        self.l0 = l0 = TstL0CacheBuffer(pspec)
        pi = l0.l0.dports[0]
        LDSTCompUnit.__init__(self, pi, rwid, 4)

    def elaborate(self, platform):
        m = LDSTCompUnit.elaborate(self, platform)
        m.submodules.l0 = self.l0
        # link addr-go direct to rel
        m.d.comb += self.ad.go_i.eq(self.ad.rel_o)
        return m


def test_scoreboard_mmu():

    units = {}
    pspec = TestMemPspec(ldst_ifacetype='mmu_cache_wb',
                         imem_ifacetype='bare_wb',
                         addr_wid=48,
                         mask_wid=8,
                         reg_wid=64,
                         units=units)

    dut = TestLDSTCompUnitMMU(16,pspec)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_ldst_comp_mmu1.il", "w") as f:
        f.write(vl)

    run_simulation(dut, ldst_sim(dut), vcd_name='test_ldst_comp.vcd')
    #TODO add wb runner here


########################################
class TestLDSTCompUnitRegSpecMMU(LDSTCompUnit):

    def __init__(self, pspec):
        from soc.experiment.l0_cache import TstL0CacheBuffer
        from soc.fu.ldst.pipe_data import LDSTPipeSpec
        regspec = LDSTPipeSpec.regspec

        # use a LoadStore1 here

        cmpi = ConfigMemoryPortInterface(pspec)
        self.cmpi = cmpi
        ldst = cmpi.pi
        self.l0 = ldst

        self.mmu = MMU()
        LDSTCompUnit.__init__(self, ldst.pi, regspec, 4)

    def elaborate(self, platform):
        m = LDSTCompUnit.elaborate(self, platform)
        m.submodules.l0 = self.l0
        m.submodules.mmu = self.mmu
        # link addr-go direct to rel
        m.d.comb += self.ad.go_i.eq(self.ad.rel_o)

        # link mmu and dcache together
        dcache = self.l0.dcache
        mmu = self.mmu
        m.d.comb += dcache.m_in.eq(mmu.d_out) # MMUToDCacheType
        m.d.comb += mmu.d_in.eq(dcache.m_out) # DCacheToMMUType

        return m

# FIXME: this is redundant code
def wb_get(wb, mem):
    """simulator process for getting memory load requests
    """

    global stop
    assert(stop==False)

    while not stop:
        while True: # wait for dc_valid
            if stop:
                return
            cyc = yield (wb.cyc)
            stb = yield (wb.stb)
            if cyc and stb:
                break
            yield
        addr = (yield wb.adr) << 3
        stop = True # hack for testing
        if addr not in mem:
            print ("    WB LOOKUP NO entry @ %x, returning zero" % (addr))

        # read or write?
        we = (yield wb.we)
        if we:
            store = (yield wb.dat_w)
            sel = (yield wb.sel)
            data = mem.get(addr, 0)
            # note we assume 8-bit sel, here
            res = 0
            for i in range(8):
                mask = 0xff << (i*8)
                if sel & (1<<i):
                    res |= store & mask
                else:
                    res |= data & mask
            mem[addr] = res
            print ("    DCACHE set %x mask %x data %x" % (addr, sel, res))
        else:
            data = mem.get(addr, 0)
            yield wb.dat_r.eq(data)
            print ("    DCACHE get %x data %x" % (addr, data))

        yield wb.ack.eq(1)
        yield
        yield wb.ack.eq(0)
        yield

def test_scoreboard_regspec_mmu():

    m = Module()

    units = {}
    pspec = TestMemPspec(ldst_ifacetype='mmu_cache_wb',
                         imem_ifacetype='bare_wb',
                         addr_wid=48,
                         mask_wid=8,
                         reg_wid=64,
                         units=units)

    dut = TestLDSTCompUnitRegSpecMMU(pspec)

    m.submodules.dut = dut

    sim = Simulator(m)
    sim.add_clock(1e-6)

    mem = pagetables.test1

    sim.add_sync_process(wrap(ldst_sim(dut)))
    sim.add_sync_process(wrap(wb_get(dut.cmpi.wb_bus(), mem)))
    with sim.write_vcd('test_scoreboard_regspec_mmu'):
        sim.run()


if __name__ == '__main__':
    #FIXME: avoid using global variables
    global stop
    stop = False
    test_scoreboard_regspec_mmu()
    #only one test for now -- test_scoreboard_mmu()
