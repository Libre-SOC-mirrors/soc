from nmigen.compat.sim import run_simulation
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Cat, Array, Const, Record, Elaboratable
from nmigen.lib.coding import Decoder

from .shadow_fn import ShadowFn


class RegDecode(Elaboratable):
    """ decodes registers into unary

        Inputs

        * :wid:         register file width
    """
    def __init__(self, wid):
        self.reg_width = wid

        # inputs
        self.enable_i = Signal(reset_less=True) # enable decoders
        self.dest_i = Signal(max=wid, reset_less=True) # Dest R# in
        self.src1_i = Signal(max=wid, reset_less=True) # oper1 R# in
        self.src2_i = Signal(max=wid, reset_less=True) # oper2 R# in

        # outputs
        self.dest_o = Signal(wid, reset_less=True) # Dest unary out
        self.src1_o = Signal(wid, reset_less=True) # oper1 unary out
        self.src2_o = Signal(wid, reset_less=True) # oper2 unary out

    def elaborate(self, platform):
        m = Module()
        m.submodules.dest_d = dest_d = Decoder(self.reg_width)
        m.submodules.src1_d = src1_d = Decoder(self.reg_width)
        m.submodules.src2_d = src2_d = Decoder(self.reg_width)

        # dest decoder: write-pending
        for d, i, o in [(dest_d, self.dest_i, self.dest_o),
                     (src1_d, self.src1_i, self.src1_o),
                     (src2_d, self.src2_i, self.src2_o)]:
            m.d.comb += d.i.eq(i)
            m.d.comb += d.n.eq(~self.enable_i)
            m.d.comb += o.eq(d.o)

        return m

    def __iter__(self):
        yield self.enable_i
        yield self.dest_i
        yield self.src1_i
        yield self.src2_i
        yield self.dest_o
        yield self.src1_o
        yield self.src2_o

    def ports(self):
        return list(self)


class IssueUnit(Elaboratable):
    """ implements 11.4.14 issue unit, p50

        Inputs

        * :wid:         register file width
        * :n_insns:     number of instructions in this issue unit.
    """
    def __init__(self, wid, n_insns):
        self.reg_width = wid
        self.n_insns = n_insns

        # inputs
        self.store_i = Signal(reset_less=True) # instruction is a store
        self.dest_i = Signal(wid, reset_less=True) # Dest R in (unary)

        self.g_wr_pend_i = Signal(wid, reset_less=True) # write pending vector

        self.insn_i = Signal(n_insns, reset_less=True, name="insn_i")
        self.busy_i = Signal(n_insns, reset_less=True, name="busy_i")

        # outputs
        self.fn_issue_o = Signal(n_insns, reset_less=True, name="fn_issue_o")
        self.g_issue_o = Signal(reset_less=True)

    def elaborate(self, platform):
        m = Module()

        if self.n_insns == 0:
            return m

        # temporaries
        waw_stall = Signal(reset_less=True)
        fu_stall = Signal(reset_less=True)
        pend = Signal(self.reg_width, reset_less=True)

        # dest decoder: write-pending
        m.d.comb += pend.eq(self.dest_i & self.g_wr_pend_i)
        m.d.comb += waw_stall.eq(pend.bool() & (~self.store_i))

        ib_l = []
        for i in range(self.n_insns):
            ib_l.append(self.insn_i[i] & self.busy_i[i])
        m.d.comb += fu_stall.eq(Cat(*ib_l).bool())
        m.d.comb += self.g_issue_o.eq(~(waw_stall | fu_stall))
        for i in range(self.n_insns):
            m.d.comb += self.fn_issue_o[i].eq(self.g_issue_o & self.insn_i[i])

        return m

    def __iter__(self):
        yield self.store_i
        yield self.dest_i
        yield self.src1_i
        yield self.src2_i
        yield self.reg_enable_i
        yield self.g_wr_pend_i
        yield from self.insn_i
        yield from self.busy_i
        yield from self.fn_issue_o
        yield self.g_issue_o

    def ports(self):
        return list(self)


class IntFPIssueUnit(Elaboratable):
    def __init__(self, wid, n_int_insns, n_fp_insns):
        self.i = IssueUnit(wid, n_int_insns)
        self.f = IssueUnit(wid, n_fp_insns)
        self.issue_o = Signal(reset_less=True)

        # some renames
        self.int_wr_pend_i = self.i.g_wr_pend_i
        self.fp_wr_pend_i = self.f.g_wr_pend_i
        self.int_wr_pend_i.name = 'int_wr_pend_i'
        self.fp_wr_pend_i.name = 'fp_wr_pend_i'

    def elaborate(self, platform):
        m = Module()
        m.submodules.intissue = self.i
        m.submodules.fpissue = self.f

        m.d.comb += self.issue_o.eq(self.i.g_issue_o | self.f.g_issue_o)

        return m

    def ports(self):
        yield self.issue_o
        yield from self.i
        yield from self.f


def issue_unit_sim(dut):
    yield dut.dest_i.eq(1)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.src1_i.eq(1)
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

def test_issue_unit():
    dut = IssueUnit(32, 3)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_issue_unit.il", "w") as f:
        f.write(vl)

    dut = IntFPIssueUnit(32, 3, 3)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_intfp_issue_unit.il", "w") as f:
        f.write(vl)

    run_simulation(dut, issue_unit_sim(dut), vcd_name='test_issue_unit.vcd')

if __name__ == '__main__':
    test_issue_unit()
