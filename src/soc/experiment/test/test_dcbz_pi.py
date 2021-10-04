"""DCache PortInterface Test
    starting as a copy to test_ldst_pi.py
"""

from nmigen import (C, Module, Signal, Elaboratable, Mux, Cat, Repl, Signal)
from nmigen.cli import main
from nmigen.cli import rtlil
from nmutil.mask import Mask, masked
from nmutil.util import Display
from random import randint, seed
from nmigen.sim import Simulator, Delay, Settle
from nmutil.util import wrap

from soc.config.test.test_pi2ls import pi_ld, pi_st, pi_ldst
from soc.config.test.test_loadstore import TestMemPspec
from soc.config.loadstore import ConfigMemoryPortInterface

from soc.fu.ldst.loadstore import LoadStore1
from soc.experiment.mmu import MMU
from soc.experiment.test import pagetables

from nmigen.compat.sim import run_simulation



stop = False

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

def setup_mmu():

    global stop
    stop = False

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
    wb_out, wb_in = dcache.wb_out, dcache.wb_in

    # link mmu and dcache together
    m.d.comb += dcache.m_in.eq(mmu.d_out) # MMUToDCacheType
    m.d.comb += mmu.d_in.eq(dcache.m_out) # DCacheToMMUType

    # link ldst and MMU together
    comb += l_in.eq(ldst.m_out)
    comb += ldst.m_in.eq(l_out)

    return m, cmpi

### test case for dcbz

def _test_dcbz_addr_100e0(dut, mem):
    mmu = dut.submodules.mmu
    pi = dut.submodules.ldst.pi
    global stop
    stop = False

    yield mmu.rin.prtbl.eq(0x1000000) # set process table
    yield

    addr = 0x100e0
    data = 0xf553b658ba7e1f51

    yield from pi_st(pi, addr, data, 8, msr_pr=0)
    yield

    ld_data = yield from pi_ld(pi, addr, 8, msr_pr=0)
    assert ld_data == 0xf553b658ba7e1f51
    ld_data = yield from pi_ld(pi, addr, 8, msr_pr=0)
    assert ld_data == 0xf553b658ba7e1f51

    print("do_dcbz ===============")
    yield from pi_st(pi, addr, data, 8, msr_pr=0, is_dcbz=1)
    print("done_dcbz ===============")
    yield

    ld_data = yield from pi_ld(pi, addr, 8, msr_pr=0)
    print("ld_data after dcbz")
    print(ld_data)
    assert ld_data == 0

    yield
    stop = True

def test_dcbz_addr_100e0():

    m, cmpi = setup_mmu()

    mem = pagetables.test1

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    sim.add_sync_process(wrap(_test_dcbz_addr_100e0(m, mem)))
    sim.add_sync_process(wrap(wb_get(cmpi.wb_bus(), mem)))
    with sim.write_vcd('test_dcbz_addr_zero.vcd'):
        sim.run()

if __name__ == '__main__':
    test_dcbz_addr_100e0()
