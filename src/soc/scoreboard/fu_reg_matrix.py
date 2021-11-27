# (DO NOT REMOVE THESE NOTICES)
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2019, 2020, 2021 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Part of the Libre-SOC Project.
# Sponsored by NLnet       EU Grant No: 825310 and 825322
# Sponsored by NGI POINTER EU Grant No: 871528

"""Mitch Alsup 6600 Dependency Matrices: Function Units to Registers (FU-REGs)

 6600 Dependency Table Matrix inputs / outputs
 ---------------------------------------------

                d s1 s2 i  d s1 s2 i  d s1 s2 i  d s1 s2 i
                | |   | |  | |   | |  | |   | |  | |   | |
                v v   v v  v v   v v  v v   v v  v v   v v
 go_rd/go_wr -> dm-r0-fu0  dm-r1-fu0  dm-r2-fu0  dm-r3-fu0 -> wr/rd-pend
 go_rd/go_wr -> dm-r0-fu1  dm-r1-fu1  dm-r2-fu1  dm-r3-fu1 -> wr/rd-pend
 go_rd/go_wr -> dm-r0-fu2  dm-r1-fu2  dm-r2-fu2  dm-r3-fu2 -> wr/rd-pend
                 |  |  |    |  |  |    |  |  |    |  |  |
                 v  v  v    v  v  v    v  v  v    v  v  v
                 d  s1 s2   d  s1 s2   d  s1 s2   d  s1 s2
                 reg sel    reg sel    reg sel    reg sel

Sub-module allocation:

                <----------- DependenceRow dr_fu0 -------> FU_RW_Pend fu_fu_0
                <----------- DependenceRow dr_fu1 -------> FU_RW_Pend fu_fu_1
                <----------- DependenceRow dr_fu2 -------> FU_RW_Pend fu_fu_2
                 |  |  |    |  |  |    |  |  |    |  |  |
                 v  v  v    v  v  v    v  v  v    v  v  v
                 Reg_Rsv    Reg_Rsv    Reg_Rsv    Reg_Rsv
                 rr_r0      rr_r1      rr_r2      rr_r3
                 |  |       |  |       |  |       |  |
                <---------- GlobalPending rd_v --------->
                <---------- GlobalPending wr_v --------->
"""

from nmigen.compat.sim import run_simulation
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Elaboratable, Cat, Repl

from soc.scoreboard.dependence_cell import DependencyRow
from soc.scoreboard.fu_wr_pending import FU_RW_Pend
from soc.scoreboard.reg_select import Reg_Rsv
from soc.scoreboard.global_pending import GlobalPending


class FURegDepMatrix(Elaboratable):
    """ implements 11.4.7 mitch alsup FU-to-Reg Dependency Matrix, p26
    """
    def __init__(self, n_fu_row, n_reg_col, n_src, n_dst, cancel=None):
        self.n_src = n_src
        self.n_dst = n_dst
        self.n_fu_row = nf = n_fu_row      # Y (FUs)   ^v
        self.n_reg_col = n_reg = n_reg_col   # X (Regs)  <>

        # arrays
        src = []
        rsel = []
        pend = []
        for i in range(n_src):
            j = i + 1 # name numbering to match src1/src2
            src.append(Signal(n_reg, name="src%d" % j, reset_less=True))
            rsel.append(Signal(n_reg, name="src%d_rsel_o" % j, reset_less=True))
            pend.append(Signal(nf, name="rd_src%d_pend_o" % j, reset_less=True))
        dst = []
        dsel = []
        dpnd = []
        for i in range(n_dst):
            j = i + 1 # name numbering to match dst1/dst2
            dst.append(Signal(n_reg, name="dst%d" % j, reset_less=True))
            dsel.append(Signal(n_reg, name="dst%d_rsel_o" % j, reset_less=True))
            dpnd.append(Signal(nf, name="wr_dst%d_pend_o" % j, reset_less=True))

        self.dst_i = tuple(dst)                              # Dest in (top)
        self.src_i = tuple(src)                              # oper in (top)
        self.dest_i = self.dst_i[0] # old API

        # cancellation array (from Address Matching), ties in with go_die_i
        self.cancel = cancel

        # Register "Global" vectors for determining RaW and WaR hazards
        self.wr_pend_i = Signal(n_reg_col, reset_less=True) # wr pending (top)
        self.rd_pend_i = Signal(n_reg_col, reset_less=True) # rd pending (top)
        self.v_wr_rsel_o = Signal(n_reg_col, reset_less=True) # wr pending (bot)
        self.v_rd_rsel_o = Signal(n_reg_col, reset_less=True) # rd pending (bot)

        self.issue_i = Signal(n_fu_row, reset_less=True)  # Issue in (top)
        self.go_wr_i = Signal(n_fu_row, reset_less=True)  # Go Write in (left)
        self.go_rd_i = Signal(n_fu_row, reset_less=True)  # Go Read in (left)
        self.go_die_i = Signal(n_fu_row, reset_less=True) # Go Die in (left)

        # for Register File Select Lines (horizontal), per-reg
        self.dst_rsel_o = tuple(dsel)                         # dest reg (bot)
        self.src_rsel_o = tuple(rsel)                         # src reg (bot)
        self.dest_rsel_o = self.dst_rsel_o[0] # old API

        # for Function Unit "forward progress" (vertical), per-FU
        self.wr_pend_o = Signal(n_fu_row, reset_less=True) # wr pending (right)
        self.rd_pend_o = Signal(n_fu_row, reset_less=True) # rd pending (right)
        self.rd_src_pend_o = tuple(pend) # src pending
        self.wr_dst_pend_o = tuple(dpnd) # dest pending

    def elaborate(self, platform):
        m = Module()
        return self._elaborate(m, platform)

    def _elaborate(self, m, platform):

        # ---
        # matrix of dependency cells.  horizontal object, allocated vertically
        # ---
        cancel_mode = self.cancel is not None
        dm = tuple(DependencyRow(self.n_reg_col, self.n_src, self.n_dst,
                                 cancel_mode=cancel_mode) \
                    for r in range(self.n_fu_row))
        for fu, dc in enumerate(dm):
            m.submodules["dr_fu%d" % fu] = dc

        # ---
        # array of Function Unit Pending vecs. allocated vertically (per FU)
        # ---
        fupend = tuple(FU_RW_Pend(self.n_reg_col, self.n_src, self.n_dst) \
                        for f in range(self.n_fu_row))
        for fu, fup in enumerate(fupend):
            m.submodules["fu_fu%d" % (fu)] = fup

        # ---
        # array of Register Reservation vecs.  allocated horizontally (per reg)
        # ---
        regrsv = tuple(Reg_Rsv(self.n_fu_row, self.n_src, self.n_dst) \
                        for r in range(self.n_reg_col))
        for rn in range(self.n_reg_col):
            m.submodules["rr_r%d" % (rn)] = regrsv[rn]

        # ---
        # connect Function Unit vector
        # ---
        wr_pend = []
        rd_pend = []
        for fup in fupend:
            # accumulate FU Vector outputs
            wr_pend.append(fup.reg_wr_pend_o)
            rd_pend.append(fup.reg_rd_pend_o)

        # ... and output them from this module (vertical, width=FUs)
        m.d.comb += self.wr_pend_o.eq(Cat(*wr_pend))
        m.d.comb += self.rd_pend_o.eq(Cat(*rd_pend))

        # connect dst fwd vectors
        for i in range(self.n_dst):
            wr_dst_pend = []
            for dc, fup in zip(dm, fupend):
                dst_fwd_o = []
                for rn in range(self.n_reg_col):
                    # accumulate cell fwd outputs for dest
                    dst_fwd_o.append(dc.dst_fwd_o[i][rn])
                # connect cell fwd outputs to FU Vector in [Cat is gooood]
                m.d.comb += fup.dst_fwd_i[i].eq(Cat(*dst_fwd_o))
                # accumulate FU Vector outputs
                wr_dst_pend.append(fup.reg_wr_dst_pend_o[i])
            # ... and output them from this module (vertical, width=FUs)
            m.d.comb += self.wr_dst_pend_o[i].eq(Cat(*wr_dst_pend))

        # same for src
        for i in range(self.n_src):
            rd_src_pend = []
            for dc, fup in zip(dm, fupend):
                src_fwd_o = []
                for rn in range(self.n_reg_col):
                    # accumulate cell fwd outputs for dest/src1/src2
                    src_fwd_o.append(dc.src_fwd_o[i][rn])
                # connect cell fwd outputs to FU Vector in [Cat is gooood]
                m.d.comb += fup.src_fwd_i[i].eq(Cat(*src_fwd_o))
                # accumulate FU Vector outputs
                rd_src_pend.append(fup.reg_rd_src_pend_o[i])
            # ... and output them from this module (vertical, width=FUs)
            m.d.comb += self.rd_src_pend_o[i].eq(Cat(*rd_src_pend))

        # ---
        # connect Reg Selection vector
        # ---
        for i in range(self.n_dst):
            dest_rsel = []
            for rn, rsv in enumerate(regrsv):
                dst_rsel_o = []
                # accumulate cell reg-select outputs dest1/2/...
                for dc in dm:
                    dst_rsel_o.append(dc.dst_rsel_o[i][rn])
                # connect cell reg-select outputs to Reg Vector In
                m.d.comb += rsv.dst_rsel_i[i].eq(Cat(*dst_rsel_o)),
                # accumulate Reg-Sel Vector outputs
                dest_rsel.append(rsv.dst_rsel_o[i])
            # ... and output them from this module (horizontal, width=REGs)
            m.d.comb += self.dst_rsel_o[i].eq(Cat(*dest_rsel))

        # same for src
        for i in range(self.n_src):
            src_rsel = []
            for rn, rsv in enumerate(regrsv):
                src_rsel_o = []
                # accumulate cell reg-select outputs src1/src2
                for dc in dm:
                    src_rsel_o.append(dc.src_rsel_o[i][rn])
                # connect cell reg-select outputs to Reg Vector In
                m.d.comb += rsv.src_rsel_i[i].eq(Cat(*src_rsel_o)),
                # accumulate Reg-Sel Vector outputs
                src_rsel.append(rsv.src_rsel_o[i])
            # ... and output them from this module (horizontal, width=REGs)
            m.d.comb += self.src_rsel_o[i].eq(Cat(*src_rsel))

        # ---
        # connect Dependency Matrix dest/src1/src2/issue to module d/s/s/i
        # ---
        for dc in dm:
            # wire up inputs from module to row cell inputs (Cat is gooood)
            m.d.comb += [dc.rd_pend_i.eq(self.rd_pend_i),
                         dc.wr_pend_i.eq(self.wr_pend_i),
                        ]
            # for dest: wire up output from module to row cell outputs
            for i in range(self.n_dst):
                m.d.comb += dc.dst_i[i].eq(self.dst_i[i])
            # for src: wire up inputs from module to row cell inputs
            for i in range(self.n_src):
                m.d.comb += dc.src_i[i].eq(self.src_i[i])

        # accumulate rsel bits into read/write pending vectors.
        rd_pend_v = []
        wr_pend_v = []
        for dc in dm:
            rd_pend_v.append(dc.v_rd_rsel_o)
            wr_pend_v.append(dc.v_wr_rsel_o)
        rd_v = GlobalPending(self.n_reg_col, rd_pend_v)
        wr_v = GlobalPending(self.n_reg_col, wr_pend_v)
        m.submodules.rd_v = rd_v
        m.submodules.wr_v = wr_v

        m.d.comb += self.v_rd_rsel_o.eq(rd_v.g_pend_o)
        m.d.comb += self.v_wr_rsel_o.eq(wr_v.g_pend_o)

        # ---
        # connect Dep issue_i/go_rd_i/go_wr_i to module issue_i/go_rd/go_wr
        # ---
        go_rd_i = []
        go_wr_i = []
        issue_i = []
        for dc in dm:
            # accumulate cell fwd outputs for dest/src1/src2
            go_rd_i.append(dc.go_rd_i)
            go_wr_i.append(dc.go_wr_i)
            issue_i.append(dc.issue_i)
        # wire up inputs from module to row cell inputs (Cat is gooood)
        m.d.comb += [Cat(*go_rd_i).eq(self.go_rd_i),
                     Cat(*go_wr_i).eq(self.go_wr_i),
                     Cat(*issue_i).eq(self.issue_i),
                    ]

        # ---
        # connect Dep go_die_i
        # ---
        if cancel_mode:
            for fu, dc in enumerate(dm):
                go_die = Repl(self.go_die_i[fu], self.n_fu_row)
                go_die = go_die | self.cancel[fu]
                m.d.comb += dc.go_die_i.eq(go_die)
        else:
            go_die_i = []
            for dc in dm:
                # accumulate cell fwd outputs for dest/src1/src2
                go_die_i.append(dc.go_die_i)
            # wire up inputs from module to row cell inputs (Cat is gooood)
            m.d.comb += Cat(*go_die_i).eq(self.go_die_i)
        return m

    def __iter__(self):
        if self.cancel is not None:
            yield self.cancel
        yield self.dest_i
        yield from self.src_i
        yield self.issue_i
        yield self.go_wr_i
        yield self.go_rd_i
        yield self.go_die_i
        yield from self.dst_rsel_o
        yield from self.src_rsel_o
        yield self.wr_pend_o
        yield self.rd_pend_o
        yield self.wr_pend_i
        yield self.rd_pend_i
        yield self.v_wr_rsel_o
        yield self.v_rd_rsel_o
        yield from self.rd_src_pend_o
        yield from self.wr_dst_pend_o

    def ports(self):
        return list(self)

def d_matrix_sim(dut):
    """ XXX TODO
    """
    yield dut.dest_i.eq(1)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.src_i[0].eq(1)
    yield dut.issue_i.eq(1)
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

def test_d_matrix():
    cancel = Signal(3)
    dut = FURegDepMatrix(n_fu_row=3, n_reg_col=4, n_src=2, n_dst=2,
                         cancel=cancel)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_fu_reg_matrix.il", "w") as f:
        f.write(vl)

    run_simulation(dut, d_matrix_sim(dut), vcd_name='test_fu_reg_matrix.vcd')

if __name__ == '__main__':
    test_d_matrix()
