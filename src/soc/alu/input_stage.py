from nmigen import (Module, Signal, Cat, Const, Mux, Repl, signed,
                    unsigned)
from nmutil.pipemodbase import PipeModBase
from soc.alu.pipe_data import ALUInitialData


class ALUInputStage(PipeModBase):
    def __init__(self, pspec):
        super().__init__(pspec, "input")

    def ispec(self):
        return ALUInitialData(self.pspec)

    def ospec(self):
        return ALUInitialData(self.pspec)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        comb += self.o.op.eq(self.i.op)

        a = Signal.like(self.i.a)

        with m.If(self.i.op.invert_a):
            comb += a.eq(~self.i.a)
        with m.Else():
            comb += a.eq(self.i.a)

        comb += self.o.a.eq(a)

        comb += self.o.b.eq(self.i.b)

        return m
