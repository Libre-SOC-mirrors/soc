from nmigen import Signal, Module, Record
from nmigen.back.pysim import Simulator, Delay
from nmigen.compat.sim import run_simulation, Settle
from nmutil.formaltest import FHDLTestCase
from nmigen.cli import rtlil
import unittest
from soc.config.test.test_loadstore import TestMemPspec
from soc.config.loadstore import ConfigMemoryPortInterface


def wait_busy(port, no=False, debug=None):
    cnt = 0
    while True:
        busy = yield port.busy_o
        print("busy", no, busy, cnt, debug)
        if bool(busy) == no:
            break
        yield
        cnt += 1


def wait_addr(port,debug=None):
    cnt = 0
    while True:
        addr_ok = yield port.addr_ok_o
        print("addrok", addr_ok,cnt,debug)
        if addr_ok:
            break
        yield
        cnt += 1


def wait_ldok(port):
    cnt = 0
    while True:
        ldok = yield port.ld.ok
        exc_happened = yield port.exc_o.happened
        print("ldok", ldok, "exception", exc_happened, "count", cnt)
        cnt += 1
        if ldok or exc_happened:
            break
        yield


def pi_st(port1, addr, data, datalen, msr_pr=0, is_dcbz=0):

    # have to wait until not busy
    yield from wait_busy(port1,debug="pi_st_A") # wait while busy

    # set up a ST on the port.  address first:
    yield port1.is_dcbz_i.eq(is_dcbz)  # reset dcbz too
    yield port1.is_st_i.eq(1)  # indicate ST
    yield port1.data_len.eq(datalen)  # ST length (1/2/4/8)
    yield port1.msr_pr.eq(msr_pr)  # MSR PR bit (1==>virt, 0==>real)

    yield port1.addr.data.eq(addr)  # set address
    yield port1.addr.ok.eq(1)  # set ok
    yield Settle()
    yield from wait_addr(port1)             # wait until addr ok
    yield from wait_addr(port1)             # wait until addr ok

    # yield # not needed, just for checking
    # yield # not needed, just for checking
    # assert "ST" for one cycle (required by the API)
    yield port1.st.data.eq(data)
    yield port1.st.ok.eq(1)
    yield
    yield port1.st.ok.eq(0)
    exc_happened = yield port1.exc_o.happened
    if exc_happened:
        print("print fast exception happened")
        yield port1.is_st_i.eq(0)  # end
        yield port1.addr.ok.eq(0)  # set !ok
        yield port1.is_dcbz_i.eq(0)  # reset dcbz too
        return "fast"
    yield from wait_busy(port1,debug="pi_st_E") # wait while busy

    # can go straight to reset.
    yield port1.is_st_i.eq(0)  # end
    yield port1.addr.ok.eq(0)  # set !ok
    yield port1.is_dcbz_i.eq(0)  # reset dcbz too
    yield  # needed if mmu/dache is used

    return None


# copy of pi_st removed

def pi_ld(port1, addr, datalen, msr_pr=0):

    # have to wait until not busy
    yield from wait_busy(port1,debug="pi_ld_A") # wait while busy

    # set up a LD on the port.  address first:
    yield port1.is_ld_i.eq(1)  # indicate LD
    yield port1.data_len.eq(datalen)  # LD length (1/2/4/8)
    yield port1.msr_pr.eq(msr_pr)  # MSR PR bit (1==>virt, 0==>real)

    yield port1.addr.data.eq(addr)  # set address
    yield port1.addr.ok.eq(1)  # set ok
    yield Settle()
    yield from wait_addr(port1)             # wait until addr ok
    exc_happened = yield port1.exc_o.happened
    if exc_happened:
        print("print fast exception happened")
        yield port1.is_ld_i.eq(0)  # end
        yield port1.addr.ok.eq(0)  # set !ok
        return None, "fast"

    yield
    yield from wait_ldok(port1)             # wait until ld ok
    data = yield port1.ld.data
    exc_happened = yield port1.exc_o.happened

    # cleanup
    yield port1.is_ld_i.eq(0)  # end
    yield port1.addr.ok.eq(0)  # set !ok
    if exc_happened:
        return None, "slow"

    yield from wait_busy(port1,debug="pi_ld_E") # wait while busy

    exc_happened = yield port1.exc_o.happened
    if exc_happened:
        return None, "slow"

    return data, None


def pi_ldst(arg, dut, msr_pr=0):

    # do two half-word stores at consecutive addresses, then two loads
    addr1 = 0x04
    addr2 = addr1 + 0x2
    data = 0xbeef
    data2 = 0xf00f
    #data = 0x4
    assert(yield from pi_st(dut, addr1, data, 2, msr_pr) is None)
    assert(yield from pi_st(dut, addr2, data2, 2, msr_pr) is None)
    result, exc = yield from pi_ld(dut, addr1, 2, msr_pr)
    result2, exc2 = yield from pi_ld(dut, addr2, 2, msr_pr)
    assert(exc is None)
    assert(exc2 is None)
    arg.assertEqual(data, result, "data %x != %x" % (result, data))
    arg.assertEqual(data2, result2, "data2 %x != %x" % (result2, data2))

    # now load both in a 32-bit load to make sure they're really consecutive
    data3 = data | (data2 << 16)
    result3, exc3 = yield from pi_ld(dut, addr1, 4, msr_pr)
    assert(exc3 is None)
    arg.assertEqual(data3, result3, "data3 %x != %x" % (result3, data3))


def tst_config_pi(testcls, ifacetype):
    """set up a configureable memory test of type ifacetype
    """
    dut = Module()
    pspec = TestMemPspec(ldst_ifacetype=ifacetype,
                         imem_ifacetype='',
                         addr_wid=48,
                         mask_wid=8,
                         reg_wid=64)
    cmpi = ConfigMemoryPortInterface(pspec)
    dut.submodules.pi = cmpi.pi
    if hasattr(cmpi, 'lsmem'):  # hmmm not happy about this
        dut.submodules.lsmem = cmpi.lsmem.lsi
    vl = rtlil.convert(dut, ports=[])  # dut.ports())
    with open("test_pi_%s.il" % ifacetype, "w") as f:
        f.write(vl)

    run_simulation(dut, {"sync": pi_ldst(testcls, cmpi.pi.pi)},
                   vcd_name='test_pi_%s.vcd' % ifacetype)


class TestPIMem(unittest.TestCase):

    def test_pi_mem(self):
        tst_config_pi(self, 'testpi')

    def test_pi2ls(self):
        tst_config_pi(self, 'testmem')

    def test_pi2ls_bare_wb(self):
        tst_config_pi(self, 'test_bare_wb')


if __name__ == '__main__':
    unittest.main()
