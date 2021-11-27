# (DO NOT REMOVE THESE NOTICES)
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2019, 2020, 2021 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Part of the Libre-SOC Project.
# Sponsored by NLnet       EU Grant No: 825310 and 825322
# Sponsored by NGI POINTER EU Grant No: 871528

from nmigen import Elaboratable, Module, Signal
from nmigen.cli import verilog, rtlil


class FU_RW_Pend(Elaboratable):
    """ these are allocated per-FU (horizontally),
        and are of length reg_count
    """
    def __init__(self, reg_count, n_src, n_dst):
        self.n_src = n_src
        self.n_dst = n_dst
        self.reg_count = reg_count
        # create dest forwarding array
        dst = []
        for i in range(n_dst):
            j = i + 1 # name numbering to match dst1/dst2
            dst.append(Signal(reg_count, name="dst%d" % j, reset_less=True))
        self.dst_fwd_i = tuple(dst)
        self.dest_fwd_i = self.dst_fwd_i[0] # old API
        # create src forwarding array
        src = []
        for i in range(n_src):
            j = i + 1 # name numbering to match src1/src2
            src.append(Signal(reg_count, name="src%d" % j, reset_less=True))
        self.src_fwd_i = tuple(src)

        self.reg_wr_pend_o = Signal(reset_less=True)
        self.reg_rd_pend_o = Signal(reset_less=True)
        self.reg_rd_src_pend_o = Signal(n_src, reset_less=True)
        self.reg_wr_dst_pend_o = Signal(n_dst, reset_less=True)

    def elaborate(self, platform):
        m = Module()
        for i in range(self.n_dst):
            m.d.comb += self.reg_wr_dst_pend_o[i].eq(self.dst_fwd_i[i].bool())
        m.d.comb += self.reg_wr_pend_o.eq(self.reg_wr_dst_pend_o.bool())
        for i in range(self.n_src):
            m.d.comb += self.reg_rd_src_pend_o[i].eq(self.src_fwd_i[i].bool())
        m.d.comb += self.reg_rd_pend_o.eq(self.reg_rd_src_pend_o.bool())
        return m

    def __iter__(self):
        yield self.reg_wr_pend_o
        yield self.reg_rd_pend_o
        yield self.reg_rd_src_pend_o
        yield self.reg_wr_dst_pend_o
        yield from self.dst_fwd_i
        yield from self.src_fwd_i

    def ports(self):
        return list(self)

def test_fu_rw_pend():
    dut = FU_RW_Pend(4, 2, 2)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_fu_rw_pend.il", "w") as f:
        f.write(vl)

if __name__ == '__main__':
    test_fu_rw_pend()
