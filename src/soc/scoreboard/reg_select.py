# (DO NOT REMOVE THESE NOTICES)
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2019, 2020, 2021 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Part of the Libre-SOC Project.
# Sponsored by NLnet       EU Grant No: 825310 and 825322
# Sponsored by NGI POINTER EU Grant No: 871528

from nmigen.cli import verilog, rtlil
from nmigen import Elaboratable, Module, Signal


class Reg_Rsv(Elaboratable):
    """ these are allocated per-Register (vertically),
        and are each of length fu_count
    """
    def __init__(self, fu_count, n_src, n_dst):
        self.n_src = n_src
        self.n_dst = n_dst
        self.fu_count = fu_count
        self.dst_rsel_i = tuple(Signal(fu_count, name="dst%i_rsel_i" % (i+1),
                                       reset_less=True) \
                                for i in range(n_dst))
        self.src_rsel_i = tuple(Signal(fu_count, name="src%i_rsel_i" % (i+1),
                                       reset_less=True) \
                                for i in range(n_src))
        self.dst_rsel_o = Signal(n_dst, reset_less=True)
        self.src_rsel_o = Signal(n_src, reset_less=True)

    def elaborate(self, platform):
        m = Module()
        for i in range(self.n_dst):
            m.d.comb += self.dst_rsel_o[i].eq(self.dst_rsel_i[i].bool())
        for i in range(self.n_src):
            m.d.comb += self.src_rsel_o[i].eq(self.src_rsel_i[i].bool())
        return m

    def __iter__(self):
        yield from self.dst_rsel_i
        yield from self.src_rsel_i
        yield self.dst_rsel_o
        yield self.src_rsel_o

    def ports(self):
        return list(self)


def test_reg_rsv():
    dut = Reg_Rsv(4, 2, 2)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_reg_rsv.il", "w") as f:
        f.write(vl)


if __name__ == '__main__':
    test_reg_rsv()
