from nmigen import (C, Module, Signal, Elaboratable, Mux, Cat, Repl, Signal,
                    Const)
from nmigen.cli import main
from nmigen.cli import rtlil
from nmutil.mask import Mask, masked
from nmutil.util import Display
from random import randint, seed
from nmigen.sim import Simulator, Delay, Settle
from nmutil.util import wrap

from soc.config.test.test_pi2ls import (pi_ld, pi_st, pi_ldst, wait_busy,
                                        get_exception_info)
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
from openpower.exceptions import LDSTExceptionTuple

from soc.config.test.test_fetch import read_from_addr
from openpower.decoder.power_enums import MSRSpec


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

    # add a debug status Signal: use "msg.str = "blah"
    # then toggle with yield msg.eq(0); yield msg.eq(1)
    debug_status = Signal(8, decoder=lambda _ : debug_status.str)
    m.debug_status = debug_status
    debug_status.str = ''

    return m, cmpi


def icache_read(dut,addr,priv,virt):

    icache = dut.submodules.ldst.icache
    i_in = icache.i_in
    i_out  = icache.i_out

    yield i_in.priv_mode.eq(priv)
    yield i_in.virt_mode.eq(virt)
    yield i_in.req.eq(1)
    yield i_in.nia.eq(addr)
    yield i_in.stop_mark.eq(0)

    yield i_in.req.eq(1)
    yield i_in.nia.eq(addr)
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

    return nia, insn, valid, failed


test_exceptions = True
test_dcbz = True
test_random = True


def debug(dut, msg):
    print ("set debug message", msg)
    dut.debug_status.str = msg # set the message
    yield dut.debug_status.eq(0) # trigger an update
    yield dut.debug_status.eq(1)


def _test_loadstore1_ifetch_iface(dut, mem):
    """test_loadstore1_ifetch_iface

    read in priv mode, non-virtual.  tests the FetchUnitInterface

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

    yield from debug(dut, "real mem instruction")
    # set address to 0x8, update mem[0x8] to 01234 | 0x5678<<32
    # (have to do 64-bit writes into the dictionary-memory-emulated-thing)
    addr = 8
    addr2 = 12
    expected_insn2 = 0x5678
    expected_insn = 0x1234
    mem[addr] = expected_insn | expected_insn2<<32

    yield i_in.priv_mode.eq(1)
    insn = yield from read_from_addr(icache, addr, stall=False)

    nia   = yield i_out.nia  # NO, must use FetchUnitInterface
    print ("fetched %x from addr %x" % (insn, nia))
    assert insn == expected_insn

    print("=== test loadstore instruction (2nd, real) ===")
    yield from debug(dut, "real mem 2nd (addr 0xc)")

    insn2 = yield from read_from_addr(icache, addr2, stall=False)

    nia   = yield i_out.nia  # NO, must use FetchUnitInterface
    print ("fetched %x from addr2 %x" % (insn2, nia))
    assert insn2 == expected_insn2

    print("=== test loadstore instruction (done) ===")

    yield from debug(dut, "test done")
    yield
    yield

    print ("fetched %x from addr %x" % (insn, nia))
    assert insn == expected_insn

    wbget.stop = True


def write_mem2(mem, addr, i1, i2):
    mem[addr] = i1 | i2<<32


#TODO: use fetch interface here
def lookup_virt(dut,addr):
    icache = dut.submodules.ldst.icache
    i_in = icache.i_in
    i_out  = icache.i_out
    yield i_in.priv_mode.eq(0)
    yield i_in.virt_mode.eq(1)
    yield i_in.req.eq(0)
    yield i_in.stop_mark.eq(0)

    yield icache.a_i_valid.eq(1)
    yield icache.a_pc_i.eq(addr)
    yield
    valid = yield i_out.valid
    failed = yield i_out.fetch_failed
    while not valid and not failed:
        yield
        valid = yield i_out.valid
        failed = yield i_out.fetch_failed
    yield icache.a_i_valid.eq(0)

    return valid,failed


def mmu_lookup(dut,addr):
    ldst = dut.submodules.ldst
    pi = ldst.pi
    yield from debug(dut, "instr fault "+hex(addr))
    yield ldst.priv_mode.eq(0)
    yield ldst.instr_fault.eq(1)
    yield ldst.maddr.eq(addr)
    yield
    yield ldst.instr_fault.eq(0)
    while True:
        done = yield (ldst.done)
        exc_info = yield from get_exception_info(pi.exc_o)
        if done or exc_info.happened:
            break
        yield
    yield
    assert exc_info.happened == 0 # assert just before doing the fault set zero
    yield ldst.instr_fault.eq(0)
    yield from debug(dut, "instr fault done "+hex(addr))
    yield
    yield
    yield

def _test_loadstore1_ifetch_multi(dut, mem):
    mmu = dut.submodules.mmu
    ldst = dut.submodules.ldst
    pi = ldst.pi
    icache = dut.submodules.ldst.icache
    assert wbget.stop == False

    print ("set process table")
    yield from debug(dut, "set prtble")
    yield mmu.rin.prtbl.eq(0x1000000) # set process table
    yield

    i_in = icache.i_in
    i_out  = icache.i_out
    i_m_in = icache.m_in

    # fetch instructions from multiple addresses
    # should cope with some addresses being invalid
    real_addrs = [0,4,8,0,8,4,0,0,12]
    write_mem2(mem,0,0xF0,0xF4)
    write_mem2(mem,8,0xF8,0xFC)

    yield i_in.priv_mode.eq(1)
    for addr in real_addrs:
        yield from debug(dut, "real_addr "+hex(addr))
        insn = yield from read_from_addr(icache, addr, stall=False)
        nia   = yield i_out.nia  # NO, must use FetchUnitInterface
        print ("TEST_MULTI: fetched %x from addr %x == %x" % (insn, nia,addr))
        assert insn==0xF0+addr

    # now with virtual memory enabled
    yield i_in.virt_mode.eq(1)

    virt_addrs = [0x10200,0x10204,0x10208,0x10200,
                  0x102008,0x10204,0x10200,0x10200,0x10200C]

    write_mem2(mem,0x10200,0xF8,0xFC)

    for addr in virt_addrs:
        yield from debug(dut, "virt_addr "+hex(addr))

        valid, failed = yield from lookup_virt(dut,addr)
        yield
        print("TEST_MULTI: failed=",failed) # this is reported wrong
        if failed==1: # test one first
            yield from mmu_lookup(dut,addr)
            valid, failed = yield from lookup_virt(dut,addr)
            assert(valid==1)

    wbget.stop = True


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
    yield from debug(dut, "set prtble")
    yield mmu.rin.prtbl.eq(0x1000000) # set process table
    yield

    yield from debug(dut, "real mem instruction")
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

    # miss, stalls for a bit -- this one is different here
    ##nia, insn, valid, failed = yield from icache_read(dut,addr,0,0)
    ##assert(valid==0)
    ##assert(failed==1)

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

    yield from debug(dut, "virtual instr req")
    # set address to 0x10200, update mem[] to 5678
    virt_addr = 0x10200
    real_addr = virt_addr
    expected_insn = 0x5678
    mem[real_addr] = expected_insn

    yield i_in.priv_mode.eq(0)
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

    yield from debug(dut, "instr fault")

    virt_addr = 0x10200

    yield ldst.priv_mode.eq(0)
    yield ldst.instr_fault.eq(1)
    yield ldst.maddr.eq(virt_addr)
    # still broken -- investigate
    # msr = MSRSpec(pr=?, dr=?, sf=0)
    # ld_data, exctype, exc = yield from pi_ld(pi, virt_addr, 8, msr=msr)
    yield
    yield ldst.instr_fault.eq(0)
    while True:
        done = yield (ldst.done)
        exc_info = yield from get_exception_info(pi.exc_o)
        if done or exc_info.happened:
            break
        yield
    assert exc_info.happened == 0 # assert just before doing the fault set zero
    yield ldst.instr_fault.eq(0)
    yield
    yield
    yield

    print("=== test loadstore instruction (try instruction again) ===")
    yield from debug(dut, "instr virt retry")
    # set address to 0x10200, update mem[] to 5678
    virt_addr = 0x10200
    real_addr = virt_addr
    expected_insn = 0x5678

    yield i_in.priv_mode.eq(0)
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
    """
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
    """

    ## part 4
    nia, insn, valid, failed = yield from icache_read(dut,virt_addr,0,1)

    yield from debug(dut, "test done")
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
    msr = MSRSpec(pr=1, dr=0, sf=0) # set problem-state
    ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr=msr)
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
        msr = MSRSpec(pr=0, dr=0, sf=0)
        yield from pi_st(pi, addr, data, 8, msr=msr)
        yield

        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr=msr)
        assert ld_data == 0xf553b658ba7e1f51
        assert exctype is None

        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr=msr)
        assert ld_data == 0xf553b658ba7e1f51
        assert exctype is None

        print("do_dcbz ===============")
        yield from pi_st(pi, addr, data, 8, msr=msr, is_dcbz=1)
        print("done_dcbz ===============")
        yield

        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr=msr)
        print("ld_data after dcbz")
        print(ld_data)
        assert ld_data == 0
        assert exctype is None

    if test_exceptions:
        print("=== alignment error (ld) ===")
        addr = 0xFF100e0FF
        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr=msr)
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
        exctype, exc = yield from pi_st(pi, addr,0, 8, msr=msr)
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
        ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr=msr)
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
            ld_data, exctype, exc  = yield from pi_ld(pi, addr, 8, msr=msr)
            print("ld_data[RANDOM]",ld_data,exc,addr)
            assert (exctype == None)

        for addr in addrs:
            print("== RANDOM addr ==",hex(addr))
            exc = yield from pi_st(pi, addr,0xFF*addr, 8, msr=msr)
            assert (exctype == None)

        # readback written data and compare
        for addr in addrs:
            print("== RANDOM addr ==",hex(addr))
            ld_data, exctype, exc = yield from pi_ld(pi, addr, 8, msr=msr)
            print("ld_data[RANDOM_READBACK]",ld_data,exc,addr)
            assert (exctype == None)
            assert (ld_data == 0xFF*addr)

        print("== RANDOM addr done ==")

    wbget.stop = True


def _test_loadstore1_ifetch_invalid(dut, mem):
    mmu = dut.submodules.mmu
    ldst = dut.submodules.ldst
    pi = ldst.pi
    icache = dut.submodules.ldst.icache
    wbget.stop = False

    print("=== test loadstore instruction (invalid) ===")

    i_in = icache.i_in
    i_out  = icache.i_out
    i_m_in = icache.m_in

    # first virtual memory test

    print ("set process table")
    yield from debug(dut, "set prtbl")
    yield mmu.rin.prtbl.eq(0x1000000) # set process table
    yield

    yield from debug(dut, "real mem instruction")
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
    nia   = yield i_out.nia
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
    yield from debug(dut, "virtual instr req")

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

    print("=== test invalid loadstore instruction (instruction fault) ===")

    yield from debug(dut, "instr fault (perm err expected)")
    virt_addr = 0x10200

    yield ldst.priv_mode.eq(0)
    yield ldst.instr_fault.eq(1)
    yield ldst.maddr.eq(virt_addr)
    #ld_data, exctype, exc = yield from pi_ld(pi, virt_addr, 8, msr=msr)
    yield
    yield ldst.instr_fault.eq(0)
    while True:
        done = yield (ldst.done)
        exc_info = yield from get_exception_info(pi.exc_o)
        if done or exc_info.happened:
            break
        yield
    assert exc_info.happened == 1 # different here as expected

    # TODO: work out what kind of exception occurred and check it's
    # the right one.  we *expect* it to be a permissions error because
    # the RPTE leaf node in pagetables.test2 is marked as "non-executable"
    # but we also expect instr_fault to be set because it is an instruction
    # (iside) lookup
    print ("   MMU lookup exception type?")
    for fname in LDSTExceptionTuple._fields:
        print ("   fname %20s %d" % (fname, getattr(exc_info, fname)))

    # ok now printed them out and visually inspected: check them with asserts
    assert exc_info.instr_fault == 1 # instruction fault (yes!)
    assert exc_info.perm_error == 1 # permissions (yes!)
    assert exc_info.rc_error == 0
    assert exc_info.alignment == 0
    assert exc_info.invalid == 0
    assert exc_info.segment_fault == 0
    assert exc_info.rc_error == 0

    yield from debug(dut, "test done")
    yield ldst.instr_fault.eq(0)
    yield
    yield
    yield

    wbget.stop = True


def test_loadstore1_ifetch_unit_iface():

    m, cmpi = setup_mmu()

    mem = pagetables.test1

    # set this up before passing to Simulator (which calls elaborate)
    icache = m.submodules.ldst.icache
    icache.use_fetch_interface() # this is the function which converts
                                 # to FetchUnitInterface. *including*
                                 # rewiring the Wishbone Bus to ibus

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    sim.add_sync_process(wrap(_test_loadstore1_ifetch_iface(m, mem)))
    # add two wb_get processes onto the *same* memory dictionary.
    # this shouuuld work.... cross-fingers...
    sim.add_sync_process(wrap(wb_get(cmpi.wb_bus(), mem)))
    sim.add_sync_process(wrap(wb_get(icache.ibus, mem))) # ibus not bus
    with sim.write_vcd('test_loadstore1_ifetch_iface.vcd',
                      traces=[m.debug_status]): # include extra debug
        sim.run()


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
    with sim.write_vcd('test_loadstore1_ifetch.vcd',
                      traces=[m.debug_status]): # include extra debug
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

def test_loadstore1_ifetch_invalid():
    m, cmpi = setup_mmu()

    # this is a specially-arranged page table which has the permissions
    # barred for execute on the leaf node (EAA=0x2 instead of EAA=0x3)
    mem = pagetables.test2

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    icache = m.submodules.ldst.icache
    sim.add_sync_process(wrap(_test_loadstore1_ifetch_invalid(m, mem)))
    # add two wb_get processes onto the *same* memory dictionary.
    # this shouuuld work.... cross-fingers...
    sim.add_sync_process(wrap(wb_get(cmpi.wb_bus(), mem)))
    sim.add_sync_process(wrap(wb_get(icache.bus, mem)))
    with sim.write_vcd('test_loadstore1_ifetch_invalid.vcd',
                      traces=[m.debug_status]): # include extra debug
        sim.run()

def test_loadstore1_ifetch_multi():
    m, cmpi = setup_mmu()
    wbget.stop = False

    # this is a specially-arranged page table which has the permissions
    # barred for execute on the leaf node (EAA=0x2 instead of EAA=0x3)
    mem = pagetables.test1

    # set this up before passing to Simulator (which calls elaborate)
    icache = m.submodules.ldst.icache
    icache.use_fetch_interface() # this is the function which converts
                                 # to FetchUnitInterface. *including*
                                 # rewiring the Wishbone Bus to ibus

    # nmigen Simulation
    sim = Simulator(m)
    sim.add_clock(1e-6)

    sim.add_sync_process(wrap(_test_loadstore1_ifetch_multi(m, mem)))
    # add two wb_get processes onto the *same* memory dictionary.
    # this shouuuld work.... cross-fingers...
    sim.add_sync_process(wrap(wb_get(cmpi.wb_bus(), mem)))
    sim.add_sync_process(wrap(wb_get(icache.ibus, mem))) # ibus not bus
    with sim.write_vcd('test_loadstore1_ifetch_multi.vcd',
                      traces=[m.debug_status]): # include extra debug
        sim.run()

if __name__ == '__main__':
    test_loadstore1()
    test_loadstore1_invalid()
    test_loadstore1_ifetch() #FIXME
    test_loadstore1_ifetch_invalid()
    test_loadstore1_ifetch_unit_iface() # guess: should be working
    test_loadstore1_ifetch_multi()
