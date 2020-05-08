# Proof of correctness for partitioned equal signal combiner
# Copyright (C) 2020 Michael Nolan <mtnolan2640@gmail.com>

from nmigen import Module, Signal, Elaboratable, Mux, Cat
from nmigen.asserts import Assert, AnyConst, Assume, Cover
from nmigen.test.utils import FHDLTestCase
from nmigen.cli import rtlil

from soc.alu.main_stage import ALUMainStage
from soc.alu.pipe_data import ALUPipeSpec
from soc.alu.alu_input_record import CompALUOpSubset
from soc.decoder.power_enums import InternalOp
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

        rec = CompALUOpSubset()
        recwidth = 0
        # Setup random inputs for dut.op
        for p in rec.ports():
            width = p.width
            recwidth += width
            comb += p.eq(AnyConst(width))

        pspec = ALUPipeSpec(id_wid=2, op_wid=recwidth)
        m.submodules.dut = dut = ALUMainStage(pspec)

        a = Signal(64)
        b = Signal(64)
        carry_in = Signal(64)
        so_in = Signal(64)
        comb += [dut.i.a.eq(a),
                 dut.i.b.eq(b),
                 dut.i.carry_in.eq(carry_in),
                 dut.i.so.eq(so_in),
                 a.eq(AnyConst(64)),
                 b.eq(AnyConst(64)),
                 carry_in.eq(AnyConst(1)),
                 so_in.eq(AnyConst(1))]
                      

        comb += dut.i.ctx.op.eq(rec)


        # Assert that op gets copied from the input to output
        for p in rec.ports():
            name = p.name
            rec_sig = p
            dut_sig = getattr(dut.o.ctx.op, name)
            comb += Assert(dut_sig == rec_sig)

        with m.If(rec.insn_type == InternalOp.OP_ADD):
            comb += Assert(Cat(dut.o.o, dut.o.carry_out) ==
                           (a + b + carry_in))


        return m

class GTCombinerTestCase(FHDLTestCase):
    def test_formal(self):
        module = Driver()
        self.assertFormal(module, mode="bmc", depth=4)
        self.assertFormal(module, mode="cover", depth=4)
    def test_ilang(self):
        dut = Driver()
        vl = rtlil.convert(dut, ports=[])
        with open("main_stage.il", "w") as f:
            f.write(vl)


if __name__ == '__main__':
    unittest.main()
