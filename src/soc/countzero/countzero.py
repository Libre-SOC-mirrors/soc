# https://github.com/antonblanchard/microwatt/blob/master/countzero.vhdl
from nmigen import Memory, Module, Signal, Cat, Elaboratable
from nmigen.hdl.rec import Record, Layout
from nmigen.cli import main


def or4(a, b, c, d):
    return Cat(a != 0, b != 0, c != 0, d != 0)


class IntermediateResult(Record):
    def __init__(self, name=None):
        layout = (('v16', 15),
                  ('sel_hi', 2),
                  ('is_32bit', 1),
                  ('count_right', 1))
        Record.__init__(self, Layout(layout), name=name)


class ZeroCounter(Elaboratable):
    def __init__(self):
        self.rs_i = Signal(64)
        self.count_right_i = Signal(1)
        self.is_32bit_i = Signal(1)
        self.result_o = Signal(64)

    def ports(self):
        return [self.rs_i, self.count_right_i, self.is_32bit_i, self.result_o]

    def elaborate(self, platform):
        m = Module()

        def encoder(v, right):
            """
            Return the index of the leftmost or rightmost 1 in a set of 4 bits.
            Assumes v is not "0000"; if it is, return (right ? "11" : "00").
            """
            ret = Signal(2)
            with m.If(right):
                with m.If(v[0]):
                    m.d.comb += ret.eq(0)
                with m.Elif(v[1]):
                    m.d.comb += ret.eq(1)
                with m.Elif(v[2]):
                    m.d.comb += ret.eq(2)
                with m.Else():
                    m.d.comb += ret.eq(3)
            with m.Else():
                with m.If(v[0]):
                    m.d.comb += ret.eq(0)
                with m.Elif(v[1]):
                    m.d.comb += ret.eq(1)
                with m.Elif(v[2]):
                    m.d.comb += ret.eq(2)
                with m.Else():
                    m.d.comb += ret.eq(3)
            return ret

        r = IntermediateResult()
        r_in = IntermediateResult()

        m.d.sync += r.eq(r_in)

        v = IntermediateResult()
        y = Signal(4)
        z = Signal(4)
        sel = Signal(6)
        v4 = Signal(4)

        # Test 4 groups of 16 bits each.
        # The top 2 groups are considered to be zero in 32-bit mode.
        m.d.comb += z.eq(or4(self.rs_i[0:16], self.rs_i[16:32],
                             self.rs_i[32:48], self.rs_i[48:64]))
        with m.If(self.is_32bit_i):
            m.d.comb += v.sel_hi[1].eq(0)
            with m.If(self.count_right_i):
                m.d.comb += v.sel_hi[0].eq(~z[0])
            with m.Else():
                m.d.comb += v.sel_hi[0].eq(z[1])
        with m.Else():
            m.d.comb += v.sel_hi.eq(encoder(z, self.count_right_i))

        # Select the leftmost/rightmost non-zero group of 16 bits

        with m.Switch(v.sel_hi):
            with m.Case(0):
                m.d.comb += v.v16.eq(self.rs_i[0:16])
            with m.Case(1):
                m.d.comb += v.v16.eq(self.rs_i[16:32])
            with m.Case(2):
                m.d.comb += v.v16.eq(self.rs_i[32:48])
            with m.Case(3):
                m.d.comb += v.v16.eq(self.rs_i[48:64])

        # Latch this and do the rest in the next cycle, for the sake of timing
        m.d.comb += v.is_32bit.eq(self.is_32bit_i)
        m.d.comb += v.count_right.eq(self.count_right_i)
        m.d.comb += r_in.eq(v)
        m.d.comb += sel[4:6].eq(r.sel_hi)

        # Test 4 groups of 4 bits
        m.d.comb += y.eq(or4(r.v16[0:4], r.v16[4:8],
                             r.v16[8:12], r.v16[12:16]))
        m.d.comb += sel[2:4].eq(encoder(y, r.count_right))

        # Select the leftmost/rightmost non-zero group of 4 bits
        with m.Switch(sel[2:4]):
            with m.Case(0):
                m.d.comb += v4.eq(r.v16[0:4])
            with m.Case(1):
                m.d.comb += v4.eq(r.v16[4:8])
            with m.Case(2):
                m.d.comb += v4.eq(r.v16[8:12])
            with m.Case(3):
                m.d.comb += v4.eq(r.v16[12:16])

        m.d.comb += sel[0:2].eq(encoder(v4, r.count_right))

        # sel is now the index of the leftmost/rightmost 1 bit in rs

        with m.If(v4 == 0):
            # operand is zero, return 32 for 32-bit, else 64
            with m.If(r.is_32bit):
                m.d.comb += self.result_o.eq(32)
            with m.Else():
                m.d.comb += self.result_o.eq(64)
        with m.Elif(r.count_right):
            # return (63 - sel), trimmed to 5 bits in 32-bit mode
            m.d.comb += self.result_o.eq(
                Cat((~sel[5] & ~r.is_32bit), ~sel[0:5]))
        with m.Else():
            m.d.comb += self.result_o.eq(sel)

        return m
