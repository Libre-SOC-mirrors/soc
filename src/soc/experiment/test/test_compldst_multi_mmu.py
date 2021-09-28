# test case for LOAD / STORE Computation Unit using MMU

from nmigen.compat.sim import run_simulation
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Mux, Cat, Elaboratable, Array, Repl
from nmigen.hdl.rec import Record, Layout

from nmutil.latch import SRLatch, latchregister
from nmutil.byterev import byte_reverse
from nmutil.extend import exts
from soc.fu.regspec import RegSpecAPI

from openpower.decoder.power_enums import MicrOp, Function, LDSTMode
from soc.fu.ldst.ldst_input_record import CompLDSTOpSubset
from openpower.decoder.power_decoder2 import Data
from openpower.consts import MSR

from soc.experiment.compalu_multi import go_record, CompUnitRecord
from soc.experiment.l0_cache import PortInterface
from soc.experiment.pimem import LDSTException
from soc.experiment.compldst_multi import LDSTCompUnit
from soc.config.test.test_loadstore import TestMemPspec

from soc.experiment.mmu import MMU

########################################
# copied from compldst_multi.py
# for debugging -- remove once done
def load_part(dut, src1, src2, imm, imm_ok=True, update=False, zero_a=False,
         byterev=True):
    print("LD_part", src1, src2, imm, imm_ok, update)
    yield dut.oper_i.insn_type.eq(MicrOp.OP_LOAD)
    yield dut.oper_i.data_len.eq(2)  # half-word
    yield dut.oper_i.byte_reverse.eq(byterev)
    yield dut.src1_i.eq(src1)
    yield dut.src2_i.eq(src2)
    yield dut.oper_i.zero_a.eq(zero_a)
    yield dut.oper_i.imm_data.imm.eq(imm)
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
        yield dut.rd.go.eq(rd)
        yield from wait_for(dut.rd.rel_o)
        yield dut.rd.go.eq(0)

# if RA = 0 then b <- 0     RA needs to be read if RA = 0
# else           b <-(RA)
# EA <- b + (RB)            RB needs to be read
# verify that EA is correct first
def dcbz(dut, ra, ra_needed, rb):
    print("LD_part", ra, ra_needed, rb)
    yield dut.oper_i.insn_type.eq(MicrOp.OP_DCBZ)
    #yield dut.oper_i.data_len.eq(2)  # half-word
    #yield dut.oper_i.byte_reverse.eq(byterev)
    yield dut.src1_i.eq(ra)
    yield dut.src2_i.eq(rb)
    #???yield dut.oper_i.zero_a.eq(zero_a)
    #yield dut.oper_i.imm_data.imm.eq(imm)
    #yield dut.oper_i.imm_data.ok.eq(imm_ok)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield


def ldst_sim(dut):
    yield from dcbz(dut, 4, True, 3) # EA=7
    #yield from load_part(dut, 4, 0, 2)
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

########################################
class TestLDSTCompUnitRegSpecMMU(LDSTCompUnit):

    def __init__(self, pspec):
        from soc.experiment.l0_cache import TstL0CacheBuffer
        from soc.fu.ldst.pipe_data import LDSTPipeSpec
        regspec = LDSTPipeSpec.regspec
        self.l0 = l0 = TstL0CacheBuffer(pspec)
        self.mmu = MMU()
        pi = l0.l0.dports[0]
        LDSTCompUnit.__init__(self, pi, regspec, 4)

    def elaborate(self, platform):
        m = LDSTCompUnit.elaborate(self, platform)
        m.submodules.l0 = self.l0
        m.submodules.mmu = self.mmu
        # link addr-go direct to rel
        m.d.comb += self.ad.go_i.eq(self.ad.rel_o)

        # link mmu and dcache together
        dcache = self.l0.pimem.dcache
        mmu = self.mmu
        m.d.comb += dcache.m_in.eq(mmu.d_out) # MMUToDCacheType
        m.d.comb += mmu.d_in.eq(dcache.m_out) # DCacheToMMUType


        return m


def test_scoreboard_regspec_mmu():

    units = {}
    pspec = TestMemPspec(ldst_ifacetype='mmu_cache_wb',
                         imem_ifacetype='bare_wb',
                         addr_wid=48,
                         mask_wid=8,
                         reg_wid=64,
                         units=units)

    dut = TestLDSTCompUnitRegSpecMMU(pspec)

    # TODO: setup pagetables for MMU

    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_ldst_comp_mmu2.il", "w") as f:
        f.write(vl)

    run_simulation(dut, ldst_sim(dut), vcd_name='test_ldst_regspec.vcd')


if __name__ == '__main__':
    test_scoreboard_regspec_mmu()
    #only one test for now -- test_scoreboard_mmu()
