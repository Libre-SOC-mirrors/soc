# Proof of correctness for partitioned equal signal combiner
# Copyright (C) 2020 Michael Nolan <mtnolan2640@gmail.com>

from nmigen import (Module, Signal, Elaboratable, Mux, Cat, Repl,
                    signed)
from nmigen.asserts import Assert, AnyConst, Assume, Cover
from nmigen.test.utils import FHDLTestCase
from nmigen.cli import rtlil

from soc.fu.logical.main_stage import LogicalMainStage
from soc.fu.alu.pipe_data import ALUPipeSpec
from soc.fu.alu.alu_input_record import CompALUOpSubset
from soc.decoder.power_enums import InternalOp
import unittest


# This defines a module to drive the device under test and assert
# properties about its outputs
class Driver(Elaboratable):
    def __init__(self):
        # inputs and outputs
        pass

    def popcount(self, sig, width):
        result = 0
        for i in range(width):
            result = result + sig[i]
        return result

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        rec = CompALUOpSubset()
        recwidth = 0
        # Setup random inputs for dut.op
        for p in rec.ports():
            width = p.width
            recwidth += width
            comb += p.eq(AnyConst(width))

        pspec = ALUPipeSpec(id_wid=2, op_wid=recwidth)
        m.submodules.dut = dut = LogicalMainStage(pspec)

        # convenience variables
        a = dut.i.a
        b = dut.i.b
        carry_in = dut.i.carry_in
        so_in = dut.i.so
        o = dut.o.o

        # setup random inputs
        comb += [a.eq(AnyConst(64)),
                 b.eq(AnyConst(64)),
                 carry_in.eq(AnyConst(1)),
                 so_in.eq(AnyConst(1))]

        comb += dut.i.ctx.op.eq(rec)

        # Assert that op gets copied from the input to output
        for rec_sig in rec.ports():
            name = rec_sig.name
            dut_sig = getattr(dut.o.ctx.op, name)
            comb += Assert(dut_sig == rec_sig)

        # signed and signed/32 versions of input a
        a_signed = Signal(signed(64))
        a_signed_32 = Signal(signed(32))
        comb += a_signed.eq(a)
        comb += a_signed_32.eq(a[0:32])

        comb += Assume(rec.insn_type == InternalOp.OP_PRTY)
        # main assertion of arithmetic operations
        with m.Switch(rec.insn_type):
            with m.Case(InternalOp.OP_AND):
                comb += Assert(dut.o.o == a & b)
            with m.Case(InternalOp.OP_OR):
                comb += Assert(dut.o.o == a | b)
            with m.Case(InternalOp.OP_XOR):
                comb += Assert(dut.o.o == a ^ b)

            with m.Case(InternalOp.OP_POPCNT):
                with m.If(rec.data_len == 8):
                    comb += Assert(dut.o.o == self.popcount(a, 64))
                with m.If(rec.data_len == 4):

                    for i in range(2):
                        comb += Assert(dut.o.o[i*32:(i+1)*32] ==
                                       self.popcount(a[i*32:(i+1)*32], 32))
                with m.If(rec.data_len == 1):
                    for i in range(8):
                        comb += Assert(dut.o.o[i*8:(i+1)*8] ==
                                       self.popcount(a[i*8:(i+1)*8], 8))

            with m.Case(InternalOp.OP_PRTY):
                with m.If(rec.data_len == 8):
                    result = 0
                    for i in range(8):
                        result = result ^ a[i*8]
                    comb += Assert(dut.o.o == result)
                with m.If(rec.data_len == 4):
                    result_low = 0
                    result_high = 0
                    for i in range(4):
                        result_low = result_low ^ a[i*8]
                        result_high = result_high ^ a[i*8 + 32]
                    comb += Assert(dut.o.o[0:32] == result_low)
                    comb += Assert(dut.o.o[32:64] == result_high)

        return m


class LogicalTestCase(FHDLTestCase):
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
