# This stage is intended to handle the gating of carry and overflow
# out, summary overflow generation, and updating the condition
# register
from nmigen import (Module, Signal, Cat, Repl)
from nmutil.pipemodbase import PipeModBase
from soc.fu.alu.pipe_data import ALUInputData, ALUOutputData
from ieee754.part.partsig import PartitionedSignal
from soc.decoder.power_enums import InternalOp


class ALUOutputStage(PipeModBase):
    def __init__(self, pspec):
        super().__init__(pspec, "output")

    def ispec(self):
        return ALUOutputData(self.pspec) # TODO: ALUIntermediateData

    def ospec(self):
        return ALUOutputData(self.pspec)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        op = self.i.ctx.op

        # op requests inversion of the output
        o = Signal.like(self.i.o)
        with m.If(op.invert_out):
            comb += o.eq(~self.i.o)
        with m.Else():
            comb += o.eq(self.i.o)

        # target register if 32-bit is only the 32 LSBs
        target = Signal(64, reset_less=True)
        with m.If(op.is_32bit):
            comb += target.eq(o[:32])
        with m.Else():
            comb += target.eq(o)

        # Handle carry_out
        with m.If(self.i.ctx.op.output_carry):
            comb += self.o.carry_out.eq(self.i.carry_out)

        # create condition register cr0 and sticky-overflow
        is_zero = Signal(reset_less=True)
        is_positive = Signal(reset_less=True)
        is_negative = Signal(reset_less=True)
        msb_test = Signal(reset_less=True) # set equal to MSB, invert if OP=CMP
        is_cmp = Signal(reset_less=True)   # true if OP=CMP
        so = Signal(reset_less=True)

        # TODO: if o[63] is XORed with "operand == OP_CMP"
        # that can be used as a test
        # see https://bugs.libre-soc.org/show_bug.cgi?id=305#c60

        comb += is_cmp.eq(op.insn_type == InternalOp.OP_CMP)
        comb += msb_test.eq(target[-1] ^ is_cmp)
        comb += is_zero.eq(target == 0)
        comb += is_positive.eq(~is_zero & ~msb_test)
        comb += is_negative.eq(~is_zero & msb_test)
        comb += so.eq(self.i.so | self.i.ov)

        with m.If(op.insn_type != InternalOp.OP_CMPEQB):
            comb += self.o.cr0.eq(Cat(so, is_zero, is_positive, is_negative))
        with m.Else():
            comb += self.o.cr0.eq(self.i.cr0)

        # copy [inverted] output, sticky-overflow and context out
        comb += self.o.o.eq(o)
        comb += self.o.so.eq(so)
        comb += self.o.ctx.eq(self.i.ctx)

        return m
