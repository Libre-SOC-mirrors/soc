from nmigen import (C, Module, Signal, Elaboratable, Mux, Cat, Repl, Signal)
from nmigen.cli import main
from nmigen.cli import rtlil
from nmutil.mask import Mask, masked
from nmutil.util import Display
from random import randint, seed
from nmigen.sim import Simulator, Delay, Settle
from nmutil.util import wrap

from soc.config.test.test_pi2ls import pi_ld, pi_st, pi_ldst, wait_busy
#from soc.config.test.test_pi2ls import pi_st_debug
from soc.config.test.test_loadstore import TestMemPspec
from soc.config.loadstore import ConfigMemoryPortInterface

from soc.fu.ldst.loadstore import LoadStore1
from soc.experiment.mmu import MMU
from soc.experiment.test import pagetables

from nmigen.compat.sim import run_simulation
from random import random

stop = False

def wb_get(wb, mem):
    """simulator process for getting memory load requests
    """

    global stop
    assert (stop==False)

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

test_exceptions = True
test_dcbz = True
test_random = True

def _test_loadstore1_invalid(dut, mem):
    mmu = dut.submodules.mmu
    pi = dut.submodules.ldst.pi
    global stop
    stop = False

    print("=== test invalid ===")

    addr = 0
    ld_data, exctype, exc, dar_o = yield from pi_ld(pi, addr, 8, msr_pr=1)
    print("ld_data", ld_data, exctype, exc)
    assert (exctype == "slow")
    invalid = exc.invalid
    assert (invalid == 1)

    print("=== test invalid done ===")

    stop = True


def _test_loadstore1(dut, mem):
    mmu = dut.submodules.mmu
    pi = dut.submodules.ldst.pi
    global stop
    stop = False

    yield mmu.rin.prtbl.eq(0x1000000) # set process table
    yield

    addr = 0x100e0
    data = 0xf553b658ba7e1f51

    if test_dcbz:
        yield from pi_st(pi, addr, data, 8, msr_pr=1)
        yield

        ld_data, exctype, exc, dar_o = yield from pi_ld(pi, addr, 8, msr_pr=1)
        assert ld_data == 0xf553b658ba7e1f51
        assert exctype is None

        ld_data, exctype, exc, dar_o  = yield from pi_ld(pi, addr, 8, msr_pr=1)
        assert ld_data == 0xf553b658ba7e1f51
        assert exctype is None

        print("do_dcbz ===============")
        yield from pi_st(pi, addr, data, 8, msr_pr=1, is_dcbz=1)
        print("done_dcbz ===============")
        yield

        ld_data, exctype, exc, dar_o  = yield from pi_ld(pi, addr, 8, msr_pr=1)
        print("ld_data after dcbz")
        print(ld_data)
        assert ld_data == 0
        assert exctype is None

    if test_exceptions:
        print("=== alignment error (ld) ===")
        addr = 0xFF100e0FF
        ld_data, exctype, exc, dar = yield from pi_ld(pi, addr, 8, msr_pr=1)
        if exc:
            alignment = exc.alignment
            happened = exc.happened
        else:
            alignment = 0
            happened = 0
        assert (happened == 1)
        assert (alignment == 1)
        assert (dar == addr)
        assert (exctype == "fast")
        yield from wait_busy(pi, debug="pi_ld_E_alignment_error")
        # wait is only needed in case of in exception here
        print("=== alignment error test passed (ld) ===")

        # take some cycles in between so that gtkwave separates out
        # signals
        yield
        yield
        yield
        yield

        print("=== alignment error (st) ===")
        addr = 0xFF100e0FF
        exctype, exc, dar_o = yield from pi_st(pi, addr,0, 8, msr_pr=1)
        if exc:
            alignment = exc.alignment
            happened = exc.happened
        else:
            alignment = 0
            happened = 0
        assert (happened == 1)
        assert (alignment==1)
        assert (dar==addr)
        assert (exctype == "fast")
        #???? yield from wait_busy(pi, debug="pi_st_E_alignment_error")
        # wait is only needed in case of in exception here
        print("=== alignment error test passed (st) ===")
        yield #FIXME hangs

    if True:
        print("=== no alignment error (ld) ===")
        addr = 0x100e0
        ld_data, exctype, exc, dar_o = yield from pi_ld(pi, addr, 8, msr_pr=1)
        print("ld_data", ld_data, exctype, exc)
        if exc:
            alignment = exc.alignment
            happened = exc.happened
        else:
            alignment = 0
            happened = 0
        assert (happened == 0)
        assert (alignment == 0)
        print("=== no alignment error done (ld) ===")

    if test_random:
        addrs = [0x456920,0xa7a180,0x299420,0x1d9d60]

        for addr in addrs:
            print("== RANDOM addr ==",hex(addr))
            ld_data, exctype, exc, dar_o  = \
                                yield from pi_ld(pi, addr, 8, msr_pr=1)
            print("ld_data[RANDOM]",ld_data,exc,addr)
            assert (exctype == None)

        for addr in addrs:
            print("== RANDOM addr ==",hex(addr))
            exc = yield from pi_st(pi, addr,0xFF*addr, 8, msr_pr=1)
            assert (exctype == None)

        # readback written data and compare
        for addr in addrs:
            print("== RANDOM addr ==",hex(addr))
            ld_data, exctype, exc, dar_o = \
                                yield from pi_ld(pi, addr, 8, msr_pr=1)
            print("ld_data[RANDOM_READBACK]",ld_data,exc,addr)
            assert (exctype == None)
            assert (ld_data == 0xFF*addr)

        print("== RANDOM addr done ==")

    stop = True

def test_loadstore1():

    m, cmpi = setup_mmu()

    mem = pagetables.test1

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    sim.add_sync_process(wrap(_test_loadstore1(m, mem)))
    sim.add_sync_process(wrap(wb_get(cmpi.wb_bus(), mem)))
    with sim.write_vcd('test_loadstore1.vcd'):
        sim.run()

def test_loadstore1_invalid():

    m, cmpi = setup_mmu()

    mem = {}

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    sim.add_sync_process(wrap(_test_loadstore1_invalid(m, mem)))
    sim.add_sync_process(wrap(wb_get(cmpi.wb_bus(), mem)))
    with sim.write_vcd('test_loadstore1_invalid.vcd'):
        sim.run()

if __name__ == '__main__':
    test_loadstore1()
    test_loadstore1_invalid()
