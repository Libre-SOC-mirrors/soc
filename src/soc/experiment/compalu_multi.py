from nmigen.compat.sim import run_simulation
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Mux, Elaboratable, Repl, Array

from nmutil.latch import SRLatch, latchregister
from soc.decoder.power_decoder2 import Data
from soc.decoder.power_enums import InternalOp

from alu_hier import CompALUOpSubset

""" Computation Unit (aka "ALU Manager").

    This module runs a "revolving door" set of three latches, based on
    * Issue
    * Go_Read
    * Go_Write
    where one of them cannot be set on any given cycle.
    (Note however that opc_l has been inverted (and qn used), due to SRLatch
     default reset state being "0" rather than "1")

    * When issue is first raised, a busy signal is sent out.
      The src1 and src2 registers and the operand can be latched in
      at this point

    * Read request is set, which is acknowledged through the Scoreboard
      to the priority picker, which generates (one and only one) Go_Read
      at a time.  One of those will (eventually) be this Computation Unit.

    * Once Go_Read is set, the src1/src2/operand latch door shuts (locking
      src1/src2/operand in place), and the ALU is told to proceed.

    * As this is currently a "demo" unit, a countdown timer is activated
      to simulate an ALU "pipeline", which activates "write request release",
      and the ALU's output is captured into a temporary register.

    * Write request release will go through a similar process as Read request,
      resulting (eventually) in Go_Write being asserted.

    * When Go_Write is asserted, two things happen: (1) the data in the temp
      register is placed combinatorially onto the output, and (2) the
      req_l latch is cleared, busy is dropped, and the Comp Unit is back
      through its revolving door to do another task.
"""


class ComputationUnitNoDelay(Elaboratable):
    def __init__(self, rwid, alu, n_src=2, n_dst=1):
        self.n_src, self.n_dst = n_src, n_dst
        self.rwid = rwid
        self.alu = alu # actual ALU - set as a "submodule" of the CU

        self.counter = Signal(4)
        src = []
        for i in range(n_src):
            j = i + 1 # name numbering to match src1/src2
            src.append(Signal(rwid, name="src%d_i" % j, reset_less=True))

        dst = []
        for i in range(n_dst):
            j = i + 1 # name numbering to match dest1/2...
            dst.append(Signal(rwid, name="dest%d_i" % j, reset_less=True))

        self.go_rd_i = Signal(n_src, reset_less=True) # read in
        self.go_wr_i = Signal(n_dst, reset_less=True) # write in
        self.issue_i = Signal(reset_less=True) # fn issue in
        self.shadown_i = Signal(reset=1) # shadow function, defaults to ON
        self.go_die_i = Signal() # go die (reset)

        # operation / data input
        self.oper_i = CompALUOpSubset() # operand
        self.src_i = Array(src)
        self.src1_i = src[0] # oper1 in
        self.src2_i = src[1] # oper2 in

        self.busy_o = Signal(reset_less=True) # fn busy out
        self.dest = Array(dst)
        self.data_o = dst[0] # Dest out
        self.rd_rel_o = Signal(n_src, reset_less=True) # release src1/src2 
        self.req_rel_o = Signal(n_dst, reset_less=True) # release out (valid_o)
        self.done_o = Signal(reset_less=True)

    def elaborate(self, platform):
        m = Module()
        m.submodules.alu = self.alu
        m.submodules.src_l = src_l = SRLatch(False, self.n_src, name="src")
        m.submodules.opc_l = opc_l = SRLatch(sync=False, name="opc")
        m.submodules.req_l = req_l = SRLatch(False, self.n_dst, name="req")
        m.submodules.rst_l = rst_l = SRLatch(sync=False, name="rst")
        m.submodules.rok_l = rok_l = SRLatch(sync=False, name="rdok")

        # ALU only proceeds when all src are ready.  rd_rel_o is delayed
        # so combine it with go_rd_i.  if all bits are set we're good
        all_rd = Signal(reset_less=True)
        m.d.comb += all_rd.eq(self.busy_o & rok_l.q &
                    (((~self.rd_rel_o) | self.go_rd_i).all()))

        # write_requests all done
        wr_any = Signal(reset_less=True)
        req_done = Signal(reset_less=True)
        m.d.comb += self.done_o.eq(self.busy_o & ~(self.req_rel_o.bool()))
        m.d.comb += wr_any.eq(self.go_wr_i.bool())
        m.d.comb += req_done.eq(self.done_o & rst_l.q & wr_any)

        # shadow/go_die
        reset = Signal(reset_less=True)
        rst_r = Signal(reset_less=True) # reset latch off
        reset_w = Signal(self.n_dst, reset_less=True)
        reset_r = Signal(self.n_src, reset_less=True)
        m.d.comb += reset.eq(req_done | self.go_die_i)
        m.d.comb += rst_r.eq(self.issue_i | self.go_die_i)
        m.d.comb += reset_w.eq(self.go_wr_i | Repl(self.go_die_i, self.n_dst))
        m.d.comb += reset_r.eq(self.go_rd_i | Repl(self.go_die_i, self.n_src))

        # read-done,wr-proceed latch
        m.d.comb += rok_l.s.eq(self.issue_i)  # set up when issue starts
        m.d.comb += rok_l.r.eq(self.alu.p_ready_o) # off when ALU acknowledges

        # wr-done, back-to-start latch
        m.d.comb += rst_l.s.eq(all_rd)     # set when read-phase is fully done
        m.d.comb += rst_l.r.eq(rst_r)        # *off* on issue

        # opcode latch (not using go_rd_i) - inverted so that busy resets to 0
        m.d.sync += opc_l.s.eq(self.issue_i)       # set on issue
        m.d.sync += opc_l.r.eq(self.alu.n_valid_o) # reset on ALU finishes

        # src operand latch (not using go_wr_i)
        m.d.sync += src_l.s.eq(Repl(self.issue_i, self.n_src))
        m.d.sync += src_l.r.eq(reset_r)

        # dest operand latch (not using issue_i)
        m.d.sync += req_l.s.eq(Repl(all_rd, self.n_dst))
        m.d.sync += req_l.r.eq(reset_w)

        # create a latch/register for the operand
        oper_r = CompALUOpSubset()
        latchregister(m, self.oper_i, oper_r, self.issue_i, "oper_r")

        # and for each output from the ALU
        drl = []
        for i in range(self.n_dst):
            name = "data_r%d" % i
            data_r = Signal(self.rwid, name=name, reset_less=True) 
            latchregister(m, self.alu.out[i], data_r, req_l.q[i], name)
            drl.append(data_r)

        # pass the operation to the ALU
        m.d.comb += self.alu.op.eq(oper_r)

        # create list of src/alu-src/src-latch.  override 2nd one below
        sl = []
        for i in range(self.n_src):
            sl.append([self.src_i[i], self.alu.i[i], src_l.q[i]])

        # select immediate if opcode says so.  however also change the latch
        # to trigger *from* the opcode latch instead.
        op_is_imm = oper_r.imm_data.imm_ok
        src2_or_imm = Signal(self.rwid, reset_less=True)
        src_sel = Signal(reset_less=True)
        m.d.comb += src_sel.eq(Mux(op_is_imm, opc_l.q, src_l.q[1]))
        m.d.comb += src2_or_imm.eq(Mux(op_is_imm, oper_r.imm_data.imm,
                                                  self.src2_i))
        # overwrite 2nd src-latch with immediate-muxed stuff
        sl[1][0] = src2_or_imm
        sl[1][2] = src_sel

        # create a latch/register for src1/src2
        for i in range(self.n_src):
            src, alusrc, latch = sl[i]
            latchregister(m, src, alusrc, latch, name="src_r%d" % i)

        # -----
        # outputs
        # -----

        # all request signals gated by busy_o.  prevents picker problems
        m.d.comb += self.busy_o.eq(opc_l.q) # busy out
        bro = Repl(self.busy_o, self.n_src)
        m.d.comb += self.rd_rel_o.eq(src_l.q & bro) # src1/src2 req rel

        # on a go_read, tell the ALU we're accepting data.
        # NOTE: this spells TROUBLE if the ALU isn't ready!
        # go_read is only valid for one clock!
        with m.If(all_rd):                           # src operands ready, GO!
            with m.If(~self.alu.p_ready_o):          # no ACK yet
                m.d.comb += self.alu.p_valid_i.eq(1) # so indicate valid

        brd = Repl(self.busy_o & self.shadown_i, self.n_dst)
        # only proceed if ALU says its output is valid
        with m.If(self.alu.n_valid_o):
            # when ALU ready, write req release out. waits for shadow
            m.d.comb += self.req_rel_o.eq(req_l.q & brd)
            # when output latch is ready, and ALU says ready, accept ALU output
            with m.If(reset):
                m.d.comb += self.alu.n_ready_i.eq(1) # tells ALU "thanks got it"

        # output the data from the latch on go_write
        for i in range(self.n_dst):
            with m.If(self.go_wr_i[i]):
                m.d.comb += self.dest[i].eq(drl[i])

        return m

    def __iter__(self):
        yield self.go_rd_i
        yield self.go_wr_i
        yield self.issue_i
        yield self.shadown_i
        yield self.go_die_i
        yield from self.oper_i.ports()
        yield self.src1_i
        yield self.src2_i
        yield self.busy_o
        yield self.rd_rel_o
        yield self.req_rel_o
        yield self.data_o

    def ports(self):
        return list(self)


def op_sim(dut, a, b, op, inv_a=0, imm=0, imm_ok=0):
    yield dut.issue_i.eq(0)
    yield
    yield dut.src_i[0].eq(a)
    yield dut.src_i[1].eq(b)
    yield dut.oper_i.insn_type.eq(op)
    yield dut.oper_i.invert_a.eq(inv_a)
    yield dut.oper_i.imm_data.imm.eq(imm)
    yield dut.oper_i.imm_data.imm_ok.eq(imm_ok)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.go_rd_i.eq(0b10)
    yield
    yield dut.go_rd_i.eq(0b01)
    while True:
        yield
        rd_rel_o = yield dut.rd_rel_o
        print ("rd_rel", rd_rel_o)
        if rd_rel_o:
            break
    yield
    yield dut.go_rd_i.eq(0)
    req_rel_o = yield dut.req_rel_o
    result = yield dut.data_o
    print ("req_rel", req_rel_o, result)
    while True:
        req_rel_o = yield dut.req_rel_o
        result = yield dut.data_o
        print ("req_rel", req_rel_o, result)
        if req_rel_o:
            break
        yield
    yield dut.go_wr_i[0].eq(1)
    yield
    result = yield dut.data_o
    print ("result", result)
    yield dut.go_wr_i[0].eq(0)
    yield
    return result


def scoreboard_sim(dut):
    result = yield from op_sim(dut, 5, 2, InternalOp.OP_ADD, inv_a=0,
                                    imm=8, imm_ok=1)
    assert result == 13

    result = yield from op_sim(dut, 5, 2, InternalOp.OP_ADD)
    assert result == 7

    result = yield from op_sim(dut, 5, 2, InternalOp.OP_ADD, inv_a=1)
    assert result == 65532


def test_scoreboard():
    from alu_hier import ALU
    from soc.decoder.power_decoder2 import Decode2ToExecute1Type

    m = Module()
    alu = ALU(16)
    dut = ComputationUnitNoDelay(16, alu)
    m.submodules.cu = dut
    run_simulation(m, scoreboard_sim(dut), vcd_name='test_compalu.vcd')

    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_compalu.il", "w") as f:
        f.write(vl)

if __name__ == '__main__':
    test_scoreboard()
