# test case for LOAD / STORE Computation Unit using MMU

from nmigen.back.pysim import Simulator, Delay, Settle, Tick
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
from soc.experiment.test.test_wishbone import wb_get

#new unit added to this test case
from soc.fu.mmu.pipe_data import MMUPipeSpec
from soc.fu.mmu.fsm import FSMMMUStage

def ldst_sim(dut):
    yield dut.mmu.rin.prtbl.eq(0x1000000) # set process table
    addr = 0x100e0
    data = 0xFF #just a single byte for this test
    #data = 0xf553b658ba7e1f51

    yield from store(dut, addr, 0, data, 0)
    yield
    ld_data, data_ok, ld_addr = yield from load(dut, addr, 0, 0)
    print(data,data_ok,ld_addr)
    assert(ld_data==data)
    yield

    data = 0

    print("doing dcbz/store with data 0 .....")
    yield from store_debug(dut, addr, 0, data, 0, dcbz=True) #hangs

    ld_data, data_ok, ld_addr = yield from load(dut, addr, 0, 0)
    print(data,data_ok,ld_addr)
    print("ld_data is")
    print(ld_data)
    assert(ld_data==data)
    print("dzbz test passed")

    dut.stop = True # stop simulation

########################################


class TestLDSTCompUnitMMUFSM(LDSTCompUnit):

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

    dut = TestLDSTCompUnit(16,pspec)
    vl = rtlil.convertMMUFSM(dut, ports=dut.ports())
    with open("test_ldst_comp_mmu1.il", "w") as f:
        f.write(vl)

    run_simulation(dut, ldst_sim(dut), vcd_name='test_ldst_comp.vcd')

########################################
class TestLDSTCompUnitRegSpecMMUFSM(LDSTCompUnit):

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

        pipe_spec = MMUPipeSpec(id_wid=2)
        self.fsm = FSMMMUStage(pipe_spec)

        self.fsm.set_ldst_interface(ldst)

        LDSTCompUnit.__init__(self, ldst.pi, regspec, 4)

    def elaborate(self, platform):
        m = LDSTCompUnit.elaborate(self, platform)
        m.submodules.l0 = self.l0
        m.submodules.mmu = self.mmu
        m.submodules.fsm = self.fsm
        # link addr-go direct to rel
        m.d.comb += self.ad.go_i.eq(self.ad.rel_o)

        # link mmu and dcache together
        dcache = self.l0.dcache
        mmu = self.mmu
        m.d.comb += dcache.m_in.eq(mmu.d_out) # MMUToDCacheType
        m.d.comb += mmu.d_in.eq(dcache.m_out) # DCacheToMMUType

        return m

def test_scoreboard_regspec_mmufsm():

    m = Module()

    units = {}
    pspec = TestMemPspec(ldst_ifacetype='mmu_cache_wb',
                         imem_ifacetype='bare_wb',
                         addr_wid=48,
                         mask_wid=8,
                         reg_wid=64,
                         units=units)

    dut = TestLDSTCompUnitRegSpecMMUFSM(pspec)

    m.submodules.dut = dut

    sim = Simulator(m)
    sim.add_clock(1e-6)

    dut.mem = pagetables.test1
    dut.stop = False

    sim.add_sync_process(wrap(ldst_sim(dut))) # rename ?
    sim.add_sync_process(wrap(wb_get(dut)))
    with sim.write_vcd('test_scoreboard_regspec_mmufsm.vcd'):
        sim.run()


if __name__ == '__main__':
    test_scoreboard_regspec_mmufsm()
    #only one test for now -- test_scoreboard_mmu()
