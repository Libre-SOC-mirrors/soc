from nmigen import (C, Module, Signal, Elaboratable, Mux, Cat, Repl, Signal,
                    Const)
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
from openpower.test.wb_get import wb_get
from openpower.test import wb_get as wbget


def setup_mmu():

    wbget.stop = False

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
    icache = ldst.icache

    l_in, l_out = mmu.l_in, mmu.l_out
    d_in, d_out = dcache.d_in, dcache.d_out
    i_in, i_out = icache.i_in, icache.i_out # FetchToICache, ICacheToDecode

    # link mmu, dcache and icache together
    m.d.comb += dcache.m_in.eq(mmu.d_out) # MMUToDCacheType
    m.d.comb += icache.m_in.eq(mmu.i_out) # MMUToICacheType
    m.d.comb += mmu.d_in.eq(dcache.m_out) # DCacheToMMUType

    # link ldst and MMU together
    comb += l_in.eq(ldst.m_out)
    comb += ldst.m_in.eq(l_out)

    return m, cmpi


test_exceptions = True
test_dcbz = True
test_random = True


def _test_loadstore1_ifetch(dut, mem):
    """test_loadstore1_ifetch

    this is quite a complex multi-step test.

    * first (just because, as a demo) read in priv mode, non-virtual.
      just like in experiment/icache.py itself.

    * second, using the (usual) PTE for these things (which came originally
      from gem5-experimental experiment/radix_walk_example.txt) do a
      virtual-memory read through the *instruction* cache.
      this is expected to FAIL

    * third: mess about with the MMU, setting "iside" (instruction-side),
      requesting an MMU RADIX LOOKUP.  this triggers an itlb_load
      (instruction-cache TLB entry-insertion)

    * fourth and finally: retry the read of the instruction through i-cache.
      this is now expected to SUCCEED

    a lot going on.
    """

    mmu = dut.submodules.mmu
    ldst = dut.submodules.ldst
    pi = ldst.pi
    icache = dut.submodules.ldst.icache
    wbget.stop = False

    print("=== test loadstore instruction (real) ===")

    i_in = icache.i_in
    i_out  = icache.i_out
    i_m_in = icache.m_in

    # first virtual memory test

    print ("set process table")
    yield mmu.rin.prtbl.eq(0x1000000) # set process table
    yield

    # set address to zero, update mem[0] to 01234
    addr = 8
    expected_insn = 0x1234
    mem[addr] = expected_insn

    yield i_in.priv_mode.eq(1)
    yield i_in.req.eq(0)
    yield i_in.nia.eq(addr)
    yield i_in.stop_mark.eq(0)
    yield i_m_in.tlbld.eq(0)
    yield i_m_in.tlbie.eq(0)
    yield i_m_in.addr.eq(0)
    yield i_m_in.pte.eq(0)
    yield
    yield
    yield

    # miss, stalls for a bit
    yield i_in.req.eq(1)
    yield i_in.nia.eq(addr)
    yield
    valid = yield i_out.valid
    while not valid:
        yield
        valid = yield i_out.valid
    yield i_in.req.eq(0)

    nia   = yield i_out.nia
    insn  = yield i_out.insn
    yield
    yield

    print ("fetched %x from addr %x" % (insn, nia))
    assert insn == expected_insn

    print("=== test loadstore instruction (virtual) ===")

    # look up i-cache expecting it to fail

    # set address to 0x10200, update mem[] to 5678
    virt_addr = 0x10200
    real_addr = virt_addr
    expected_insn = 0x5678
    mem[real_addr] = expected_insn

    yield i_in.priv_mode.eq(1)
    yield i_in.virt_mode.eq(1)
    yield i_in.req.eq(0)
    yield i_in.nia.eq(virt_addr)
    yield i_in.stop_mark.eq(0)
    yield i_m_in.tlbld.eq(0)
    yield i_m_in.tlbie.eq(0)
    yield i_m_in.addr.eq(0)
    yield i_m_in.pte.eq(0)
    yield
    yield
    yield

    # miss, stalls for a bit
    yield i_in.req.eq(1)
    yield i_in.nia.eq(virt_addr)
    yield
    valid = yield i_out.valid
    failed = yield i_out.fetch_failed
    while not valid and not failed:
        yield
        valid = yield i_out.valid
        failed = yield i_out.fetch_failed
    yield i_in.req.eq(0)

    print ("failed?", "yes" if failed else "no")
    assert failed == 1
    yield
    yield

    print("=== test loadstore instruction (instruction fault) ===")

    virt_addr = 0x10200

    yield ldst.priv_mode.eq(1)
    yield ldst.instr_fault.eq(1)
    yield ldst.maddr.eq(virt_addr)
    #ld_data, exctype, exc = yield from pi_ld(pi, virt_addr, 8, msr_pr=1)
    yield
    yield ldst.instr_fault.eq(0)
    while True:
        done = yield (ldst.done)
        if done:
            break
        yield
    yield ldst.instr_fault.eq(0)
    yield
    yield
    yield

    print("=== test loadstore instruction (try instruction again) ===")
    # set address to 0x10200, update mem[] to 5678
    virt_addr = 0x10200
    real_addr = virt_addr
    expected_insn = 0x5678

    yield i_in.priv_mode.eq(1)
    yield i_in.virt_mode.eq(1)
    yield i_in.req.eq(0)
    yield i_in.nia.eq(virt_addr)
    yield i_in.stop_mark.eq(0)
    yield i_m_in.tlbld.eq(0)
    yield i_m_in.tlbie.eq(0)
    yield i_m_in.addr.eq(0)
    yield i_m_in.pte.eq(0)
    yield
    yield
    yield

    # miss, stalls for a bit
    yield i_in.req.eq(1)
    yield i_in.nia.eq(virt_addr)
    yield
    valid = yield i_out.valid
    failed = yield i_out.fetch_failed
    while not valid and not failed:
        yield
        valid = yield i_out.valid
        failed = yield i_out.fetch_failed
    yield i_in.req.eq(0)
    nia   = yield i_out.nia
    insn  = yield i_out.insn
    yield
    yield

    print ("failed?", "yes" if failed else "no")
    assert failed == 0

    print ("fetched %x from addr %x" % (insn, nia))
    assert insn == expected_insn

    wbget.stop = True


def _test_loadstore1_invalid(dut, mem):
    mmu = dut.submodules.mmu
    pi = dut.submodules.ldst.pi
    wbget.stop = False

    print("=== test invalid ===")

    addr = 0
    ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr_pr=1)
    print("ld_data", ld_data, exctype, exc)
    assert (exctype == "slow")
    invalid = exc.invalid
    assert (invalid == 1)

    print("=== test invalid done ===")

    wbget.stop = True


def _test_loadstore1(dut, mem):
    mmu = dut.submodules.mmu
    pi = dut.submodules.ldst.pi
    ldst = dut.submodules.ldst # to get at DAR (NOT part of PortInterface)
    wbget.stop = False

    yield mmu.rin.prtbl.eq(0x1000000) # set process table
    yield

    addr = 0x100e0
    data = 0xf553b658ba7e1f51

    if test_dcbz:
        yield from pi_st(pi, addr, data, 8, msr_pr=1)
        yield

        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr_pr=1)
        assert ld_data == 0xf553b658ba7e1f51
        assert exctype is None

        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr_pr=1)
        assert ld_data == 0xf553b658ba7e1f51
        assert exctype is None

        print("do_dcbz ===============")
        yield from pi_st(pi, addr, data, 8, msr_pr=1, is_dcbz=1)
        print("done_dcbz ===============")
        yield

        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr_pr=1)
        print("ld_data after dcbz")
        print(ld_data)
        assert ld_data == 0
        assert exctype is None

    if test_exceptions:
        print("=== alignment error (ld) ===")
        addr = 0xFF100e0FF
        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr_pr=1)
        if exc:
            alignment = exc.alignment
            happened = exc.happened
            yield # wait for dsr to update
            dar = yield ldst.dar
        else:
            alignment = 0
            happened = 0
            dar = 0
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
        exctype, exc = yield from pi_st(pi, addr,0, 8, msr_pr=1)
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
        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr_pr=1)
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
            ld_data, exctype, exc  = yield from pi_ld(pi, addr, 8, msr_pr=1)
            print("ld_data[RANDOM]",ld_data,exc,addr)
            assert (exctype == None)

        for addr in addrs:
            print("== RANDOM addr ==",hex(addr))
            exc = yield from pi_st(pi, addr,0xFF*addr, 8, msr_pr=1)
            assert (exctype == None)

        # readback written data and compare
        for addr in addrs:
            print("== RANDOM addr ==",hex(addr))
            ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr_pr=1)
            print("ld_data[RANDOM_READBACK]",ld_data,exc,addr)
            assert (exctype == None)
            assert (ld_data == 0xFF*addr)

        print("== RANDOM addr done ==")

    wbget.stop = True


def test_loadstore1_ifetch():

    m, cmpi = setup_mmu()

    mem = pagetables.test1

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    icache = m.submodules.ldst.icache
    sim.add_sync_process(wrap(_test_loadstore1_ifetch(m, mem)))
    # add two wb_get processes onto the *same* memory dictionary.
    # this shouuuld work.... cross-fingers...
    sim.add_sync_process(wrap(wb_get(cmpi.wb_bus(), mem)))
    sim.add_sync_process(wrap(wb_get(icache.bus, mem)))
    with sim.write_vcd('test_loadstore1_ifetch.vcd'):
        sim.run()


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
    test_loadstore1_ifetch()
