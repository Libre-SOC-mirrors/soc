# This stage is intended to adjust the input data before sending it to
# the acutal ALU. Things like handling inverting the input, carry_in
# generation for subtraction, should happen here
from nmigen import (Module, Signal)
from nmutil.pipemodbase import PipeModBase
from soc.decoder.power_enums import MicrOp
from soc.decoder.power_enums import CryIn


class CommonInputStage(PipeModBase):

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        op = self.i.ctx.op

        ##### operand A #####

        # operand a to be as-is or inverted
        a = Signal.like(self.i.a)

        if hasattr(op, "invert_a"):
            with m.If(op.invert_a):
                comb += a.eq(~self.i.a)
            with m.Else():
                comb += a.eq(self.i.a)
        else:
            comb += a.eq(self.i.a)

        comb += self.o.a.eq(a)

        ##### carry-in #####

        # either copy incoming carry or set to 1/0 as defined by op
        if hasattr(self.i, "xer_ca"): # hack (for now - for LogicalInputData)
            with m.Switch(op.input_carry):
                with m.Case(CryIn.ZERO):
                    comb += self.o.xer_ca.eq(0b00)
                with m.Case(CryIn.ONE):
                    comb += self.o.xer_ca.eq(0b11) # XER CA/CA32
                with m.Case(CryIn.CA):
                    comb += self.o.xer_ca.eq(self.i.xer_ca)

        ##### sticky overflow and context (both pass-through) #####

        if hasattr(self.o, "xer_so"): # hack (for now - for LogicalInputData)
            with m.If(op.oe.oe_ok):
                comb += self.o.xer_so.eq(self.i.xer_so)
        comb += self.o.ctx.eq(self.i.ctx)

        return m
