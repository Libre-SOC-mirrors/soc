# Proof of correctness for shift/rotate FU
# Copyright (C) 2020 Michael Nolan <mtnolan2640@gmail.com>
"""
Links:
* https://bugs.libre-soc.org/show_bug.cgi?id=340
"""

import unittest
import enum
from nmigen import (Module, Signal, Elaboratable, Mux, Cat, Repl,
                    signed, Const, unsigned)
from nmigen.asserts import Assert, AnyConst, Assume
from nmutil.formaltest import FHDLTestCase
from nmutil.sim_util import do_sim
from nmigen.sim import Delay

from soc.fu.shift_rot.main_stage import ShiftRotMainStage
from soc.fu.shift_rot.pipe_data import ShiftRotPipeSpec
from openpower.decoder.power_enums import MicrOp


@enum.unique
class TstOp(enum.Enum):
    """ops we're testing, the idea is if we run a separate formal proof for
    each instruction, we end up covering them all and each runs much faster,
    also the formal proofs can be run in parallel."""
    SHL = MicrOp.OP_SHL
    SHR = MicrOp.OP_SHR
    RLC32 = MicrOp.OP_RLC, 32
    RLC64 = MicrOp.OP_RLC, 64
    RLCL = MicrOp.OP_RLCL
    RLCR = MicrOp.OP_RLCR
    EXTSWSLI = MicrOp.OP_EXTSWSLI
    TERNLOG = MicrOp.OP_TERNLOG
    GREV32 = MicrOp.OP_GREV, 32
    GREV64 = MicrOp.OP_GREV, 64

    @property
    def op(self):
        if isinstance(self.value, tuple):
            return self.value[0]
        return self.value


def eq_any_const(sig: Signal):
    return sig.eq(AnyConst(sig.shape(), src_loc_at=1))


class Mask(Elaboratable):
    # copied from qemu's mask fn:
    # https://gitlab.com/qemu-project/qemu/-/blob/477c3b934a47adf7de285863f59d6e4503dd1a6d/target/ppc/internal.h#L21
    def __init__(self):
        self.start = Signal(6)
        self.end = Signal(6)
        self.out = Signal(64)

    def elaborate(self, platform):
        m = Module()
        max_val = Const(~0, unsigned(64))
        max_bit = 63
        with m.If(self.start == 0):
            m.d.comb += self.out.eq(max_val << (max_bit - self.end))
        with m.Elif(self.end == max_bit):
            m.d.comb += self.out.eq(max_val >> self.start)
        with m.Else():
            ret = (max_val >> self.start) ^ ((max_val >> self.end) >> 1)
            m.d.comb += self.out.eq(Mux(self.start > self.end, ~ret, ret))
        return m


class TstMask(unittest.TestCase):
    def test_mask(self):
        dut = Mask()

        def case(start, end, expected):
            with self.subTest(start=start, end=end):
                yield dut.start.eq(start)
                yield dut.end.eq(end)
                yield Delay(1e-6)
                out = yield dut.out
                with self.subTest(out=hex(out), expected=hex(expected)):
                    self.assertEqual(expected, out)

        def process():
            for start in range(64):
                for end in range(64):
                    expected = 0
                    if start > end:
                        for i in range(start, 64):
                            expected |= 1 << (63 - i)
                        for i in range(0, end + 1):
                            expected |= 1 << (63 - i)
                    else:
                        for i in range(start, end + 1):
                            expected |= 1 << (63 - i)
                    yield from case(start, end, expected)
        with do_sim(self, dut, [dut.start, dut.end, dut.out]) as sim:
            sim.add_process(process)
            sim.run()


def rotl64(v, amt):
    v |= Const(0, 64)  # convert to value at least 64-bits wide
    amt |= Const(0, 6)  # convert to value at least 6-bits wide
    return (Cat(v[:64], v[:64]) >> (64 - amt[:6]))[:64]


def rotl32(v, amt):
    v |= Const(0, 32)  # convert to value at least 32-bits wide
    return rotl64(Cat(v[:32], v[:32]), amt)


# This defines a module to drive the device under test and assert
# properties about its outputs
class Driver(Elaboratable):
    def __init__(self, which):
        assert isinstance(which, TstOp)
        self.which = which

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        pspec = ShiftRotPipeSpec(id_wid=2, parent_pspec=None)
        pspec.draft_bitmanip = True
        m.submodules.dut = dut = ShiftRotMainStage(pspec)

        # Set inputs to formal variables
        comb += [
            eq_any_const(dut.i.ctx.op.insn_type),
            eq_any_const(dut.i.ctx.op.fn_unit),
            eq_any_const(dut.i.ctx.op.imm_data.data),
            eq_any_const(dut.i.ctx.op.imm_data.ok),
            eq_any_const(dut.i.ctx.op.rc.rc),
            eq_any_const(dut.i.ctx.op.rc.ok),
            eq_any_const(dut.i.ctx.op.oe.oe),
            eq_any_const(dut.i.ctx.op.oe.ok),
            eq_any_const(dut.i.ctx.op.write_cr0),
            eq_any_const(dut.i.ctx.op.input_carry),
            eq_any_const(dut.i.ctx.op.output_carry),
            eq_any_const(dut.i.ctx.op.input_cr),
            eq_any_const(dut.i.ctx.op.is_32bit),
            eq_any_const(dut.i.ctx.op.is_signed),
            eq_any_const(dut.i.ctx.op.insn),
            eq_any_const(dut.i.xer_ca),
            eq_any_const(dut.i.ra),
            eq_any_const(dut.i.rb),
            eq_any_const(dut.i.rc),
        ]

        # check that the operation (op) is passed through (and muxid)
        comb += Assert(dut.o.ctx.op == dut.i.ctx.op)
        comb += Assert(dut.o.ctx.muxid == dut.i.ctx.muxid)

        # we're only checking a particular operation:
        comb += Assume(dut.i.ctx.op.insn_type == self.which.op)

        # dispatch to check fn for each op
        getattr(self, f"_check_{self.which.name.lower()}")(m, dut)

        return m

    def _check_shl(self, m, dut):
        m.d.comb += Assume(dut.i.ra == 0)
        expected = Signal(64)
        with m.If(dut.i.ctx.op.is_32bit):
            m.d.comb += expected.eq((dut.i.rs << dut.i.rb[:6])[:32])
        with m.Else():
            m.d.comb += expected.eq((dut.i.rs << dut.i.rb[:7])[:64])
        m.d.comb += Assert(dut.o.o.data == expected)
        m.d.comb += Assert(dut.o.xer_ca.data == 0)

    def _check_shr(self, m, dut):
        m.d.comb += Assume(dut.i.ra == 0)
        expected = Signal(64)
        carry = Signal()
        shift_in_s = Signal(signed(128))
        shift_roundtrip = Signal(signed(128))
        shift_in_u = Signal(128)
        shift_amt = Signal(7)
        with m.If(dut.i.ctx.op.is_32bit):
            m.d.comb += [
                shift_amt.eq(dut.i.rb[:6]),
                shift_in_s.eq(dut.i.rs[:32].as_signed()),
                shift_in_u.eq(dut.i.rs[:32]),
            ]
        with m.Else():
            m.d.comb += [
                shift_amt.eq(dut.i.rb[:7]),
                shift_in_s.eq(dut.i.rs.as_signed()),
                shift_in_u.eq(dut.i.rs),
            ]

        with m.If(dut.i.ctx.op.is_signed):
            m.d.comb += [
                expected.eq(shift_in_s >> shift_amt),
                shift_roundtrip.eq((shift_in_s >> shift_amt) << shift_amt),
                carry.eq((shift_in_s < 0) & (shift_roundtrip != shift_in_s)),
            ]
        with m.Else():
            m.d.comb += [
                expected.eq(shift_in_u >> shift_amt),
                carry.eq(0),
            ]
        m.d.comb += Assert(dut.o.o.data == expected)
        m.d.comb += Assert(dut.o.xer_ca.data == Repl(carry, 2))

    def _check_rlc32(self, m, dut):
        m.d.comb += Assume(dut.i.ctx.op.is_32bit)
        # rlwimi, rlwinm, and rlwnm

        m.submodules.mask = mask = Mask()
        expected = Signal(64)
        rot = Signal(64)
        m.d.comb += rot.eq(rotl32(dut.i.rs[:32], dut.i.rb[:5]))
        m.d.comb += mask.start.eq(dut.fields.FormM.MB[:] + 32)
        m.d.comb += mask.end.eq(dut.fields.FormM.ME[:] + 32)

        # for rlwinm and rlwnm, ra is guaranteed to be 0, so that part of
        # the expression turns into a no-op
        m.d.comb += expected.eq((rot & mask.out) | (dut.i.ra & ~mask.out))
        m.d.comb += Assert(dut.o.o.data == expected)
        m.d.comb += Assert(dut.o.xer_ca.data == 0)

    def _check_rlc64(self, m, dut):
        m.d.comb += Assume(~dut.i.ctx.op.is_32bit)
        # rldic and rldimi

        # `rb` is always a 6-bit immediate
        m.d.comb += Assume(dut.i.rb[6:] == 0)

        m.submodules.mask = mask = Mask()
        expected = Signal(64)
        rot = Signal(64)
        m.d.comb += rot.eq(rotl64(dut.i.rs, dut.i.rb[:6]))
        mb = dut.fields.FormMD.mb[:]
        m.d.comb += mask.start.eq(Cat(mb[1:6], mb[0]))
        m.d.comb += mask.end.eq(63 - dut.i.rb[:6])

        # for rldic, ra is guaranteed to be 0, so that part of
        # the expression turns into a no-op
        m.d.comb += expected.eq((rot & mask.out) | (dut.i.ra & ~mask.out))
        m.d.comb += Assert(dut.o.o.data == expected)
        m.d.comb += Assert(dut.o.xer_ca.data == 0)

    def _check_rlcl(self, m, dut):
        m.d.comb += Assume(~dut.i.ctx.op.is_32bit)
        # rldicl and rldcl

        m.d.comb += Assume(~dut.i.ctx.op.is_signed)
        m.d.comb += Assume(dut.i.ra == 0)

        m.submodules.mask = mask = Mask()
        m.d.comb += mask.end.eq(63)
        mb = dut.fields.FormMD.mb[:]
        m.d.comb += mask.start.eq(Cat(mb[1:6], mb[0]))

        rot = Signal(64)
        m.d.comb += rot.eq(rotl64(dut.i.rs, dut.i.rb[:6]))

        expected = Signal(64)
        m.d.comb += expected.eq(rot & mask.out)

        m.d.comb += Assert(dut.o.o.data == expected)
        m.d.comb += Assert(dut.o.xer_ca.data == 0)

    def _check_rlcr(self, m, dut):
        m.d.comb += Assume(~dut.i.ctx.op.is_32bit)
        # rldicr and rldcr

        m.d.comb += Assume(~dut.i.ctx.op.is_signed)
        m.d.comb += Assume(dut.i.ra == 0)

        m.submodules.mask = mask = Mask()
        m.d.comb += mask.start.eq(0)
        me = dut.fields.FormMD.me[:]
        m.d.comb += mask.end.eq(Cat(me[1:6], me[0]))

        rot = Signal(64)
        m.d.comb += rot.eq(rotl64(dut.i.rs, dut.i.rb[:6]))

        expected = Signal(64)
        m.d.comb += expected.eq(rot & mask.out)

        m.d.comb += Assert(dut.o.o.data == expected)
        m.d.comb += Assert(dut.o.xer_ca.data == 0)

    def _check_extswsli(self, m, dut):
        m.d.comb += Assume(dut.i.ra == 0)
        m.d.comb += Assume(dut.i.rb[6:] == 0)
        m.d.comb += Assume(~dut.i.ctx.op.is_32bit)  # all instrs. are 64-bit
        expected = Signal(64)
        m.d.comb += expected.eq((dut.i.rs[0:32].as_signed() << dut.i.rb[:6]))
        m.d.comb += Assert(dut.o.o.data == expected)
        m.d.comb += Assert(dut.o.xer_ca.data == 0)

    def _check_ternlog(self, m, dut):
        lut = dut.fields.FormTLI.TLI[:]
        for i in range(64):
            idx = Cat(dut.i.rb[i], dut.i.ra[i], dut.i.rc[i])
            for j in range(8):
                with m.If(j == idx):
                    m.d.comb += Assert(dut.o.o.data[i] == lut[j])
        m.d.comb += Assert(dut.o.xer_ca.data == 0)

    def _check_grev32(self, m, dut):
        m.d.comb += Assume(dut.i.ctx.op.is_32bit)
        # assert zero-extended
        m.d.comb += Assert(dut.o.o.data[32:] == 0)
        i = Signal(5)
        m.d.comb += eq_any_const(i)
        idx = dut.i.rb[0: 5] ^ i
        m.d.comb += Assert((dut.o.o.data >> i)[0] == (dut.i.ra >> idx)[0])
        m.d.comb += Assert(dut.o.xer_ca.data == 0)

    def _check_grev64(self, m, dut):
        m.d.comb += Assume(~dut.i.ctx.op.is_32bit)
        i = Signal(6)
        m.d.comb += eq_any_const(i)
        idx = dut.i.rb[0: 6] ^ i
        m.d.comb += Assert((dut.o.o.data >> i)[0] == (dut.i.ra >> idx)[0])
        m.d.comb += Assert(dut.o.xer_ca.data == 0)


class ALUTestCase(FHDLTestCase):
    def run_it(self, which):
        module = Driver(which)
        self.assertFormal(module, mode="bmc", depth=2)
        self.assertFormal(module, mode="cover", depth=2)

    def test_shl(self):
        self.run_it(TstOp.SHL)

    def test_shr(self):
        self.run_it(TstOp.SHR)

    def test_rlc32(self):
        self.run_it(TstOp.RLC32)

    def test_rlc64(self):
        self.run_it(TstOp.RLC64)

    def test_rlcl(self):
        self.run_it(TstOp.RLCL)

    def test_rlcr(self):
        self.run_it(TstOp.RLCR)

    def test_extswsli(self):
        self.run_it(TstOp.EXTSWSLI)

    def test_ternlog(self):
        self.run_it(TstOp.TERNLOG)

    def test_grev32(self):
        self.run_it(TstOp.GREV32)

    def test_grev64(self):
        self.run_it(TstOp.GREV64)


# check that all test cases are covered
for i in TstOp:
    assert callable(getattr(ALUTestCase, f"test_{i.name.lower()}"))


if __name__ == '__main__':
    unittest.main()
