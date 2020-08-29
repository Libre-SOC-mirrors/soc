# Proof of correctness for partitioned equal signal combiner
# Copyright (C) 2020 Michael Nolan <mtnolan2640@gmail.com>

from nmigen import (Module, Signal, Elaboratable, Mux, Cat, Repl,
                    signed)
from nmigen.asserts import Assert, AnyConst, Assume, Cover
from nmutil.formaltest import FHDLTestCase
from nmutil.stageapi import StageChain
from nmigen.cli import rtlil

from soc.fu.mul.pipe_data import CompMULOpSubset, MulPipeSpec
from soc.fu.mul.pre_stage import MulMainStage1
from soc.fu.mul.main_stage import MulMainStage2
from soc.fu.mul.post_stage import MulMainStage3

from soc.decoder.power_enums import MicrOp
import unittest


# This defines a module to drive the device under test and assert
# properties about its outputs
class Driver(Elaboratable):
    def __init__(self):
        # inputs and outputs
        pass

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        rec = CompMULOpSubset()

        # Setup random inputs for dut.op
        comb += rec.insn_type.eq(AnyConst(rec.insn_type.width))
        comb += rec.fn_unit.eq(AnyConst(rec.fn_unit.width))
        comb += rec.is_signed.eq(AnyConst(rec.is_signed.width))
        comb += rec.is_32bit.eq(AnyConst(rec.is_32bit.width))
        comb += rec.imm_data.imm.eq(AnyConst(64))
        comb += rec.imm_data.imm_ok.eq(AnyConst(1))
        # TODO, the rest of these.  (the for-loop hides Assert-failures)

        # set up the mul stages.  do not add them to m.submodules, this
        # is handled by StageChain.setup().
        pspec = MulPipeSpec(id_wid=2)
        pipe1 = MulMainStage1(pspec)
        pipe2 = MulMainStage2(pspec)
        pipe3 = MulMainStage3(pspec)

        class Dummy: pass
        dut = Dummy() # make a class into which dut.i and dut.o can be dropped
        dut.i = pipe1.ispec()
        chain = [pipe1, pipe2, pipe3] # chain of 3 mul stages

        StageChain(chain).setup(m, dut.i) # input linked here, through chain
        dut.o = chain[-1].o # output is the last thing in the chain...

        # convenience variables
        a = dut.i.ra
        b = dut.i.rb
        o = dut.o.o.data

        # work out absolute (as 32 bit signed) of a and b
        abs32_a = Signal(32)
        abs32_b = Signal(32)
        a32_s = Signal(1)
        b32_s = Signal(1)
        comb += a32_s.eq(a[31])
        comb += b32_s.eq(b[31])
        comb += abs32_a.eq(Mux(a32_s, -a[0:32], a[0:32]))
        comb += abs32_b.eq(Mux(b32_s, -b[0:32], b[0:32]))

        # work out absolute (as 64 bit signed) of a and b
        abs64_a = Signal(64)
        abs64_b = Signal(64)
        a64_s = Signal(1)
        b64_s = Signal(1)
        comb += a64_s.eq(a[63])
        comb += b64_s.eq(b[63])
        comb += abs64_a.eq(Mux(a64_s, -a[0:64], a[0:64]))
        comb += abs64_b.eq(Mux(b64_s, -b[0:64], b[0:64]))

        # a same sign as b
        ab32_seq = Signal()
        ab64_seq = Signal()
        comb += ab32_seq.eq(a32_s ^ b32_s)
        comb += ab64_seq.eq(a64_s ^ b64_s)

        # setup random inputs
        comb += [a.eq(AnyConst(64)),
                 b.eq(AnyConst(64)),
                ]

        comb += dut.i.ctx.op.eq(rec)

        # Assert that op gets copied from the input to output
        comb += Assert(dut.o.ctx.op == dut.i.ctx.op)
        comb += Assert(dut.o.ctx.muxid == dut.i.ctx.muxid)

        # Assert that XER_SO propagates through as well.
        # Doesn't mean that the ok signal is always set though.
        comb += Assert(dut.o.xer_so.data == dut.i.xer_so)

        # main assertion of arithmetic operations
        with m.Switch(rec.insn_type):
            with m.Case(MicrOp.OP_MUL_H32):
                comb += Assume(rec.is_32bit) # OP_MUL_H32 is a 32-bit op

                exp_prod = Signal(64)
                expected_o = Signal.like(exp_prod)

                # unsigned hi32 - mulhwu
                with m.If(~rec.is_signed):
                    comb += exp_prod.eq(a[0:32] * b[0:32])
                    comb += expected_o.eq(Repl(exp_prod[32:64], 2))
                    comb += Assert(o[0:64] == expected_o)

                # signed hi32 - mulhw
                with m.Else():
                    prod = Signal.like(exp_prod) # intermediate product
                    comb += prod.eq(abs32_a * abs32_b)
                    # TODO: comment why a[31]^b[31] is used to invert prod?
                    comb += exp_prod.eq(Mux(ab32_seq, -prod, prod))
                    comb += expected_o.eq(Repl(exp_prod[32:64], 2))
                    comb += Assert(o[0:64] == expected_o)

            with m.Case(MicrOp.OP_MUL_H64):
                comb += Assume(~rec.is_32bit)

                exp_prod = Signal(128)

                # unsigned hi64 - mulhdu
                with m.If(~rec.is_signed):
                    comb += exp_prod.eq(a[0:64] * b[0:64])
                    comb += Assert(o[0:64] == exp_prod[64:128])

                # signed hi64 - mulhd
                with m.Else():
                    prod = Signal.like(exp_prod) # intermediate product
                    comb += prod.eq(abs64_a * abs64_b)
                    comb += exp_prod.eq(Mux(ab64_seq, -prod, prod))
                    comb += Assert(o[0:64] == exp_prod[64:128])

            # mulli, mullw(o)(u), mulld(o)
            with m.Case(MicrOp.OP_MUL_L64):
                with m.If(rec.is_32bit):
                    expected_ov = Signal()
                    prod = Signal(64)
                    exp_prod = Signal.like(prod)

                    # unsigned lo32 - mullwu
                    with m.If(~rec.is_signed):
                        comb += exp_prod.eq(a[0:32] * b[0:32])
                        comb += Assert(o[0:64] == exp_prod[0:64])

                    # signed lo32 - mullw
                    with m.Else():
                        # TODO: comment why a[31]^b[31] is used to invert prod?
                        comb += prod.eq(abs32_a[0:64] * abs32_b[0:64])
                        comb += exp_prod.eq(Mux(ab32_seq, -prod, prod))
                        comb += Assert( o[0:64] == exp_prod[0:64])

                    # TODO: how does m31.bool &  ~m31.all work?
                    m31 = exp_prod[31:64]
                    comb += expected_ov.eq(m31.bool() & ~m31.all())
                    comb += Assert(dut.o.xer_ov.data == Repl(expected_ov, 2))

                with m.Else(): # is 64-bit; mulld
                    expected_ov = Signal()
                    prod = Signal(128)
                    exp_prod = Signal.like(prod)

                    # From my reading of the v3.0B ISA spec,
                    # only signed instructions exist.
                    comb += Assume(rec.is_signed)

                    # TODO: comment why a[63]^b[63] is used to invert prod?
                    comb += prod.eq(abs64_a[0:64] * abs64_b[0:64])
                    comb += exp_prod.eq(Mux(ab64_seq, -prod, prod))
                    comb += Assert(o[0:64] == exp_prod[0:64])

                    # TODO: how does m63.bool &  ~m63.all work?
                    m63 = exp_prod[63:128]
                    comb += expected_ov.eq(m63.bool() & ~m63.all())
                    comb += Assert(dut.o.xer_ov.data == Repl(expected_ov, 2))

        return m


class MulTestCase(FHDLTestCase):
    def test_formal(self):
        module = Driver()
        self.assertFormal(module, mode="bmc", depth=2)
        self.assertFormal(module, mode="cover", depth=2)
    def test_ilang(self):
        dut = Driver()
        vl = rtlil.convert(dut, ports=[])
        with open("main_stage.il", "w") as f:
            f.write(vl)


if __name__ == '__main__':
    unittest.main()
