# This stage is intended to adjust the input data before sending it to
# the acutal ALU. Things like handling inverting the input, xer_ca
# generation for subtraction, and handling of immediates should happen
# here
from nmigen import (Module, Signal, Cat, Const, Mux, Repl, signed,
                    unsigned)
from nmutil.pipemodbase import PipeModBase
from soc.decoder.power_enums import InternalOp
from soc.fu.alu.pipe_data import ALUInputData
from soc.decoder.power_enums import CryIn


class ALUInputStage(PipeModBase):
    def __init__(self, pspec):
        super().__init__(pspec, "input")

    def ispec(self):
        return ALUInputData(self.pspec)

    def ospec(self):
        return ALUInputData(self.pspec)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        ctx = self.i.ctx

        ##### operand A #####

        # operand a to be as-is or inverted
        a = Signal.like(self.i.a)

        with m.If(ctx.op.invert_a):
            comb += a.eq(~self.i.a)
        with m.Else():
            comb += a.eq(self.i.a)

        comb += self.o.a.eq(a)
        comb += self.o.b.eq(self.i.b)

        ##### carry-in #####

        # either copy incoming carry or set to 1/0 as defined by op
        with m.Switch(ctx.op.input_carry):
            with m.Case(CryIn.ZERO):
                comb += self.o.xer_ca.eq(0b00)
            with m.Case(CryIn.ONE):
                comb += self.o.xer_ca.eq(0b11) # set both CA and CA32
            with m.Case(CryIn.CA):
                comb += self.o.xer_ca.eq(self.i.xer_ca)

        ##### sticky overflow and context (both pass-through) #####

        comb += self.o.xer_so.eq(self.i.xer_so)
        comb += self.o.ctx.eq(ctx)

        return m
