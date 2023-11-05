# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2022 Cesar Strauss <cestrauss@gmail.com>
# Sponsored by NLnet under EU Grant and 957073
# Part of the Libre-SOC Project.

"""
Formal proof of soc.experiment.compalu_multi.MultiCompUnit

In short, MultiCompUnit:

1) stores an opcode from Issue, when not "busy", and "issue" is pulsed
2) signals "busy" high
3) fetches its operand(s), if any (which are not masked or zero) from the
Scoreboard (REL/GO protocol)
4) starts the ALU (ready/valid protocol), as soon as all inputs are available
5) captures result from ALU (again ready/valid)
5) sends the result(s) back to the Scoreboard (again REL/GO)
6) drops "busy"

Note that, if the conditions are right, many of the above can occur together,
on a single cycle.

The formal proof involves ensuring that:
1) the ALU gets the right opcode from Issue
2) the ALU gets the right operands from the Scoreboard
3) the Scoreboard receives the right result from the ALU
4) no transactions are dropped or repeated

This can be checked using holding registers and transaction counters.

See https://bugs.libre-soc.org/show_bug.cgi?id=879 and
https://bugs.libre-soc.org/show_bug.cgi?id=197
"""

import unittest

from nmigen import Signal, Module
from nmigen.hdl.ast import Cover, Const, Assume, Assert
from nmutil.formaltest import FHDLTestCase
from nmutil.singlepipe import ControlBase

from soc.experiment.compalu_multi import MultiCompUnit
from soc.fu.alu.alu_input_record import CompALUOpSubset


# Formal model of a simple ALU, whose inputs and outputs are randomly
# generated by the formal engine

class ALUCtx:
    def __init__(self):
        self.op = CompALUOpSubset(name="op")


class ALUInput:
    def __init__(self):
        self.a = Signal(16)
        self.b = Signal(16)
        self.ctx = ALUCtx()

    def eq(self, i):
        return [self.a.eq(i.a), self.b.eq(i.b)]


class ALUOutput:
    def __init__(self):
        self.o1 = Signal(16)
        self.o2 = Signal(16)

    def eq(self, i):
        return [self.o1.eq(i.o1), self.o2.eq(i.o2)]


class ALU(ControlBase):
    def __init__(self):
        super().__init__(stage=self)
        self.p.i_data, self.n.o_data = self.new_specs(None)
        self.i, self.o = self.p.i_data, self.n.o_data

    def setup(self, m, i):
        pass

    def ispec(self, name=None):
        return ALUInput()

    def ospec(self, name=None):
        return ALUOutput()

    def elaborate(self, platform):
        m = super().elaborate(platform)
        return m


class CompALUMultiTestCase(FHDLTestCase):
    def test_formal(self):
        inspec = [('INT', 'a', '0:15'),
                  ('INT', 'b', '0:15')]
        outspec = [('INT', 'o1', '0:15'),
                   ('INT', 'o2', '0:15')]
        regspec = (inspec, outspec)
        m = Module()
        # Instantiate "random" ALU
        alu = ALU()
        m.submodules.dut = dut = MultiCompUnit(regspec, alu, CompALUOpSubset)
        # TODO Test shadow / die
        m.d.comb += [dut.shadown_i.eq(1), dut.go_die_i.eq(0)]
        # Don't issue while busy
        issue = Signal()
        m.d.comb += dut.issue_i.eq(issue & ~dut.busy_o)
        # Avoid toggling go_i when rel_o is low (rel / go protocol)
        rd_go = Signal(dut.n_src)
        m.d.comb += dut.cu.rd.go_i.eq(rd_go & dut.cu.rd.rel_o)
        wr_go = Signal(dut.n_dst)
        m.d.comb += dut.cu.wr.go_i.eq(wr_go & dut.cu.wr.rel_o)
        # Transaction counters
        do_issue = Signal()
        m.d.comb += do_issue.eq(dut.issue_i & ~dut.busy_o)
        cnt_issue = Signal(4)
        m.d.sync += cnt_issue.eq(cnt_issue + do_issue)
        do_read = Signal(dut.n_src)
        m.d.comb += do_read.eq(dut.cu.rd.rel_o & dut.cu.rd.go_i)
        cnt_read = []
        for i in range(dut.n_src):
            cnt = Signal(4, name="cnt_read_%d" % i)
            m.d.sync += cnt.eq(cnt + do_read[i])
            cnt_read.append(cnt)
        do_write = Signal(dut.n_dst)
        m.d.comb += do_write.eq(dut.cu.wr.rel_o & dut.cu.wr.go_i)
        cnt_write = []
        for i in range(dut.n_dst):
            cnt = Signal(4, name="cnt_write_%d" % i)
            m.d.sync += cnt.eq(cnt + do_write[i])
            cnt_write.append(cnt)
        do_alu_write = Signal()
        m.d.comb += do_alu_write.eq(alu.p.i_valid & alu.p.o_ready)
        cnt_alu_write = Signal(4)
        m.d.sync += cnt_alu_write.eq(cnt_alu_write + do_alu_write)
        do_alu_read = Signal()
        m.d.comb += do_alu_read.eq(alu.n.o_valid & alu.n.i_ready)
        cnt_alu_read = Signal(4)
        m.d.sync += cnt_alu_read.eq(cnt_alu_read + do_alu_read)
        cnt_masked_read = []
        do_masked_read = Signal(dut.n_src)
        for i in range(dut.n_src):
            cnt = Signal(4, name="cnt_masked_read_%d" % i)
            if i == 0:
                extra = dut.oper_i.zero_a
            elif i == 1:
                extra = dut.oper_i.imm_data.ok
            else:
                extra = Const(0, 1)
            m.d.comb += do_masked_read[i].eq(do_issue &
                                             (dut.rdmaskn[i] | extra))
            m.d.sync += cnt.eq(cnt + do_masked_read[i])
            cnt_masked_read.append(cnt)
        # If the ALU is idle, do not assert valid
        with m.If((cnt_alu_read == cnt_alu_write) & ~do_alu_write):
            m.d.comb += Assume(~alu.n.o_valid)
        # Keep ALU valid high, until read
        last_alu_valid = Signal()
        m.d.sync += last_alu_valid.eq(alu.n.o_valid & ~alu.n.i_ready)
        with m.If(last_alu_valid):
            m.d.comb += Assume(alu.n.o_valid)

        # Invariant checks

        # For every instruction issued, at any point in time,
        # each operand was either:
        # 1) Already read
        # 2) Not read yet, but the read is pending (rel_o high)
        # 3) Masked
        for i in range(dut.n_src):
            sum_read = Signal(4)
            m.d.comb += sum_read.eq(
                cnt_read[i] + cnt_masked_read[i] + dut.cu.rd.rel_o[i])
            m.d.comb += Assert(sum_read == cnt_issue)

        # For every instruction, either:
        # 1) The ALU is executing the instruction
        # 2) Otherwise, execution is pending (alu.p.i_valid is high)
        # 3) Otherwise, it is waiting for operands
        #    (some dut.cu.rd.rel_o are still high)
        # 4) ... unless all operands are masked, in which case there is a one
        #    cycle delay
        all_masked = Signal()
        m.d.sync += all_masked.eq(do_masked_read.all())
        sum_alu_write = Signal(4)
        m.d.comb += sum_alu_write.eq(
            cnt_alu_write +
            (dut.cu.rd.rel_o.any() | all_masked | alu.p.i_valid))
        m.d.comb += Assert(sum_alu_write == cnt_issue)

        # Ask the formal engine to give an example
        m.d.comb += Cover((cnt_issue == 2)
                          & (cnt_read[0] == 1)
                          & (cnt_read[1] == 0)
                          & (cnt_write[0] == 1)
                          & (cnt_write[1] == 1)
                          & (cnt_alu_write == 1)
                          & (cnt_alu_read == 1)
                          & (cnt_masked_read[0] == 1)
                          & (cnt_masked_read[1] == 1))
        with self.subTest("cover"):
            self.assertFormal(m, mode="cover", depth=10)

        # Check assertions
        with self.subTest("bmc"):
            self.assertFormal(m, mode="bmc", depth=10)


if __name__ == "__main__":
    unittest.main()
