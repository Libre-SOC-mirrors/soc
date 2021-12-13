"""MMU PortInterface Test

quite basic, goes directly to the MMU to assert signals (does not
yet use PortInterface)
"""

from nmigen import (C, Module, Signal, Elaboratable, Mux, Cat, Repl, Signal)
from nmigen.cli import main
from nmigen.cli import rtlil
from nmutil.mask import Mask, masked
from nmutil.util import Display

if True:
    from nmigen.back.pysim import Simulator, Delay, Settle
else:
    from nmigen.sim.cxxsim import Simulator, Delay, Settle
from nmutil.util import wrap

from soc.config.test.test_pi2ls import pi_ld, pi_st, pi_ldst
from soc.config.test.test_loadstore import TestMemPspec
from soc.config.loadstore import ConfigMemoryPortInterface

from soc.fu.ldst.loadstore import LoadStore1
from soc.experiment.mmu import MMU

from nmigen.compat.sim import run_simulation
from openpower.test.wb_get import wb_get
from openpower.test import wb_get as wbget
from openpower.decoder.power_enums import MSRSpec

msr_default = MSRSpec(pr=1, dr=0, sf=1) # 64 bit by default


wbget.stop = False

def b(x): # byte-reverse function
    return int.from_bytes(x.to_bytes(8, byteorder='little'),
                          byteorder='big', signed=False)


def setup_mmu():

    pspec = TestMemPspec(ldst_ifacetype='mmu_cache_wb',
                         imem_ifacetype='',
                         addr_wid=48,
                         #disable_cache=True, # hmmm...
                         mask_wid=8,
                         reg_wid=64)

    m = Module()
    comb = m.d.comb
    cmpi = ConfigMemoryPortInterface(pspec)
    m.submodules.ldst = ldst = cmpi.pi
    m.submodules.mmu = mmu = MMU()
    dcache = ldst.dcache

    l_in, l_out = mmu.l_in, mmu.l_out
    d_in, d_out = dcache.d_in, dcache.d_out

    # link mmu and dcache together
    m.d.comb += dcache.m_in.eq(mmu.d_out) # MMUToDCacheType
    m.d.comb += mmu.d_in.eq(dcache.m_out) # DCacheToMMUType

    # link ldst and MMU together
    comb += l_in.eq(ldst.m_out)
    comb += ldst.m_in.eq(l_out)

    return m, cmpi



def ldst_sim_misalign(dut):
    mmu = dut.submodules.mmu
    wbget.stop = False

    yield mmu.rin.prtbl.eq(0x1000000) # set process table
    yield

    # load 8 bytes at aligned address
    align_addr = 0x1000
    data, exctype, exc = yield from pi_ld(dut.submodules.ldst.pi,
                                          align_addr, 8, msr=msr_default)
    print ("ldst_sim_misalign (aligned)", hex(data), exctype, exc)
    assert data == 0xdeadbeef01234567

    # load 4 bytes at aligned address
    align_addr = 0x1004
    data, exctype, exc = yield from pi_ld(dut.submodules.ldst.pi,
                                          align_addr, 4, msr=msr_default)
    print ("ldst_sim_misalign (aligned)", hex(data), exctype, exc)
    assert data == 0xdeadbeef

    # load 8 bytes at *mis*-aligned address
    misalign_addr = 0x1004
    data, exctype, exc = yield from pi_ld(dut.submodules.ldst.pi,
                                          misalign_addr, 8, msr=msr_default)
    print ("ldst_sim_misalign", data, exctype, exc)
    yield
    dar = yield dut.submodules.ldst.dar
    print ("DAR", hex(dar))
    assert dar == misalign_addr
    # check exception bits
    assert exc.happened
    assert exc.alignment
    assert not exc.segment_fault
    assert not exc.instr_fault
    assert not exc.invalid
    assert not exc.perm_error
    assert not exc.rc_error
    assert not exc.badtree

    wbget.stop = True


def test_misalign_mmu():

    m, cmpi = setup_mmu()

    # virtual "memory" to use for this test

    mem = {0x10000:    # PARTITION_TABLE_2
                       # PATB_GR=1 PRTB=0x1000 PRTS=0xb
           b(0x800000000100000b),

           0x30000:     # RADIX_ROOT_PTE
                        # V = 1 L = 0 NLB = 0x400 NLS = 9
           b(0x8000000000040009),

           0x40000:     # RADIX_SECOND_LEVEL
                        # 	   V = 1 L = 1 SW = 0 RPN = 0
                           # R = 1 C = 1 ATT = 0 EAA 0x7
           b(0xc000000000000183),

          0x1000000:   # PROCESS_TABLE_3
                       # RTS1 = 0x2 RPDB = 0x300 RTS2 = 0x5 RPDS = 13
           b(0x40000000000300ad),

         # data to return
          0x1000: 0xdeadbeef01234567,
          0x1008: 0xfeedf00ff001a5a5
          }


    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    sim.add_sync_process(wrap(ldst_sim_misalign(m)))
    sim.add_sync_process(wrap(wb_get(cmpi.wb_bus(), mem)))
    with sim.write_vcd('test_ldst_pi_misalign.vcd'):
        sim.run()


if __name__ == '__main__':
    test_misalign_mmu()
