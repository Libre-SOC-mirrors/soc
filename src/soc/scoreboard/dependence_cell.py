# (DO NOT REMOVE THESE NOTICES)
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2019, 2020, 2021 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Part of the Libre-SOC Project.
# Sponsored by NLnet       EU Grant No: 825310 and 825322
# Sponsored by NGI POINTER EU Grant No: 871528

from nmigen.compat.sim import run_simulation
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Elaboratable, Cat, Repl
from nmutil.latch import SRLatch


class DependencyRow(Elaboratable):
    """ implements 11.4.7 mitch alsup dependence cell, p27
        adjusted to be clock-sync'd on rising edge only.
        mitch design (as does 6600) requires alternating rising/falling clock

        * SET mode: issue_i HI, go_i LO, reg_i HI - register is captured
                                                  - FWD is DISABLED (~issue_i)
                                                  - RSEL DISABLED
        * QRY mode: issue_i LO, go_i LO, haz_i HI - FWD is ASSERTED
                                         reg_i HI - ignored
        * GO mode : issue_i LO, go_i HI           - RSEL is ASSERTED
                                         haz_i HI - FWD still can be ASSERTED

        FWD assertion (hazard protection) therefore still occurs in both
        Query and Go Modes, for this cycle, due to the cq register

        GO mode works for one cycle, again due to the cq register capturing
        the latch output.  Without the cq register, the SR Latch (which is
        asynchronous) would be reset at the exact moment that GO was requested,
        and the RSEL would be garbage.
    """
    def __init__(self, n_reg, n_src, n_dst, cancel_mode=False):
        self.cancel_mode = cancel_mode
        self.n_reg = n_reg
        self.n_src = n_src
        self.n_dst = n_dst
        # arrays
        src = []
        rsel = []
        fwd = []
        for i in range(n_src):
            j = i + 1 # name numbering to match src1/src2
            src.append(Signal(n_reg, name="src%d" % j, reset_less=True))
            rsel.append(Signal(n_reg, name="src%d_rsel_o" % j, reset_less=True))
            fwd.append(Signal(n_reg, name="src%d_fwd_o" % j, reset_less=True))
        dst = []
        dsel = []
        dfwd = []
        for i in range(n_dst):
            j = i + 1 # name numbering to match src1/src2
            dst.append(Signal(n_reg, name="dst%d" % j, reset_less=True))
            dsel.append(Signal(n_reg, name="dst%d_rsel_o" % j, reset_less=True))
            dfwd.append(Signal(n_reg, name="dst%d_fwd_o" % j, reset_less=True))

        # inputs
        self.dst_i = tuple(dst)                # Dest in (top)
        self.src_i = tuple(src)                # operands in (top)
        self.issue_i = Signal(reset_less=True) # Issue in (top)

        self.rd_pend_i = Signal(n_reg, reset_less=True) # Read pend in (top)
        self.wr_pend_i = Signal(n_reg, reset_less=True) # Write pend in (top)
        self.v_rd_rsel_o = Signal(n_reg, reset_less=True) # Read pend out (bot)
        self.v_wr_rsel_o = Signal(n_reg, reset_less=True) # Write pend out (bot)

        self.go_wr_i = Signal(reset_less=True) # Go Write in (left)
        self.go_rd_i = Signal(reset_less=True)  # Go Read in (left)
        if self.cancel_mode:
            self.go_die_i = Signal(n_reg, reset_less=True) # Go Die in (left)
        else:
            self.go_die_i = Signal(reset_less=True) # Go Die in (left)

        # for Register File Select Lines (vertical)
        self.dst_rsel_o = tuple(dsel)         # dest reg sel (bot)
        self.src_rsel_o = tuple(rsel)         # src reg sel (bot)

        # for Function Unit "forward progress" (horizontal)
        self.dst_fwd_o = tuple(dfwd)        # dest FU fw (right)
        self.src_fwd_o = tuple(fwd)         # src FU fw (right)

        # for temporary (transitional) compatibility with old API
        # number of dests used to be 1 (one) - increasing to n_dst
        self.dest_i = self.dst_i[0]
        self.dest_rsel_o = self.dst_rsel_o[0]
        self.dest_fwd_o = self.dst_fwd_o[0]

    def elaborate(self, platform):
        m = Module()
        # create source and dest SRLatches
        dst_c = []
        for i in range(self.n_dst):
            dst_l = SRLatch(sync=False, llen=self.n_reg)
            m.submodules["dst%d_c" % (i+1)] = dst_l
            dst_c.append(dst_l)

        src_c = []
        for i in range(self.n_src):
            src_l = SRLatch(sync=False, llen=self.n_reg)
            m.submodules["src%d_c" % (i+1)] = src_l
            src_c.append(src_l)

        # connect go_rd / go_wr (dest->wr, src->rd)
        wr_die = Signal(self.n_reg, reset_less=True)
        rd_die = Signal(self.n_reg, reset_less=True)
        if self.cancel_mode:
            go_die = self.go_die_i
        else:
            go_die = Repl(self.go_die_i, self.n_reg)
        m.d.comb += wr_die.eq(Repl(self.go_wr_i, self.n_reg) | go_die)
        m.d.comb += rd_die.eq(Repl(self.go_rd_i, self.n_reg) | go_die)
        for i in range(self.n_dst):
            m.d.comb += dst_c[i].r.eq(wr_die)
        for i in range(self.n_src):
            m.d.comb += src_c[i].r.eq(rd_die)

        # connect input reg bit (unary)
        i_ext = Repl(self.issue_i, self.n_reg)
        for i in range(self.n_dst):
            m.d.comb += dst_c[i].s.eq(i_ext & self.dst_i[i])
        for i in range(self.n_src):
            m.d.comb += src_c[i].s.eq(i_ext & self.src_i[i])

        # connect up hazard checks: read-after-write and write-after-read
        for i in range(self.n_dst):
            m.d.comb += self.dst_fwd_o[i].eq(dst_c[i].q & self.rd_pend_i)
        for i in range(self.n_src):
            m.d.comb += self.src_fwd_o[i].eq(src_c[i].q & self.wr_pend_i)

        # connect reg-sel outputs
        rd_ext = Repl(self.go_rd_i, self.n_reg)
        wr_ext = Repl(self.go_wr_i, self.n_reg)
        for i in range(self.n_dst):
            m.d.comb += self.dst_rsel_o[i].eq(dst_c[i].qlq & wr_ext)
        for i in range(self.n_src):
            m.d.comb += self.src_rsel_o[i].eq(src_c[i].qlq & rd_ext)

        # to be accumulated to indicate if register is in use (globally)
        # after ORing, is fed back in to rd_pend_i / wr_pend_i
        src_q = []
        for i in range(self.n_src):
            src_q.append(src_c[i].qlq)
        m.d.comb += self.v_rd_rsel_o.eq(Cat(*src_q).bool())
        dst_q = []
        for i in range(self.n_dst):
            dst_q.append(dst_c[i].qlq)
        m.d.comb += self.v_wr_rsel_o.eq(Cat(*dst_q).bool())

        return m

    def __iter__(self):
        yield from self.dst_i
        yield from self.src_i
        yield self.rd_pend_i
        yield self.wr_pend_i
        yield self.issue_i
        yield self.go_wr_i
        yield self.go_rd_i
        yield self.go_die_i
        yield from self.dst_rsel_o
        yield from self.src_rsel_o
        yield from self.dst_fwd_o
        yield from self.src_fwd_o

    def ports(self):
        return list(self)


# XXX not up-to-date but hey
def dcell_sim(dut):
    yield dut.dest_i.eq(1)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.src_i[0].eq(1)
    yield dut.issue_i.eq(1)
    yield
    yield
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.go_rd_i.eq(1)
    yield
    yield dut.go_rd_i.eq(0)
    yield
    yield dut.go_wr_i.eq(1)
    yield
    yield dut.go_wr_i.eq(0)
    yield

def test_dcell():
    dut = DependencyRow(4, 2, 2, True)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_drow.il", "w") as f:
        f.write(vl)

    run_simulation(dut, dcell_sim(dut), vcd_name='test_dcell.vcd')

if __name__ == '__main__':
    test_dcell()
