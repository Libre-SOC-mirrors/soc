from nmigen.compat.sim import run_simulation
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Elaboratable

from nmutil.latch import SRLatch, latchregister

""" LOAD / STORE Computation Unit.  Also capable of doing ADD and ADD immediate

    This module runs a "revolving door" set of four latches, based on
    * Issue
    * Go_Read
    * Go_Addr
    * Go_Write *OR* Go_Store

    (Note that opc_l has been inverted (and qn used), due to SRLatch
     default reset state being "0" rather than "1")
"""

# internal opcodes.  hypothetically this could do more combinations.
# meanings:
# * bit 0: 0 = ADD , 1 = SUB
# * bit 1: 0 = src1, 1 = IMM
# * bit 2: 1 = LD
# * bit 3: 1 = ST
LDST_OP_ADDI = 0b0000 # plain ADD (src1 + src2)
LDST_OP_SUBI = 0b0001 # plain SUB (src1 - src2)
LDST_OP_ADD  = 0b0010 # immed ADD (imm + src1)
LDST_OP_SUB  = 0b0011 # immed SUB (imm - src1)
LDST_OP_ST   = 0b0110 # immed ADD plus LD op.  ADD result is address
LDST_OP_LD   = 0b1010 # immed ADD plus ST op.  ADD result is address


class LDSTCompUnit(Elaboratable):
    def __init__(self, rwid, opwid, alu, mem):
        self.rwid = rwid
        self.alu = alu
        self.mem = mem

        self.counter = Signal(4)
        self.go_rd_i = Signal(reset_less=True) # go read in
        self.go_ad_i = Signal(reset_less=True) # go address in
        self.go_wr_i = Signal(reset_less=True) # go write in
        self.go_st_i = Signal(reset_less=True) # go store in
        self.issue_i = Signal(reset_less=True) # fn issue in
        self.isalu_i = Signal(reset_less=True) # fn issue as ALU in
        self.shadown_i = Signal(reset=1) # shadow function, defaults to ON
        self.go_die_i = Signal() # go die (reset)

        self.oper_i = Signal(opwid, reset_less=True) # opcode in
        self.imm_i = Signal(rwid, reset_less=True) # immediate in
        self.src1_i = Signal(rwid, reset_less=True) # oper1 in
        self.src2_i = Signal(rwid, reset_less=True) # oper2 in

        self.busy_o = Signal(reset_less=True)       # fn busy out
        self.rd_rel_o = Signal(reset_less=True) # request src1/src2
        self.adr_rel_o = Signal(reset_less=True) # request address (from mem)
        self.sto_rel_o = Signal(reset_less=True) # request store (to mem) 
        self.req_rel_o = Signal(reset_less=True) # request write (result)
        self.data_o = Signal(rwid, reset_less=True) # Dest out (LD or ALU)
        self.load_mem_o = Signal(reset_less=True) # activate memory LOAD
        self.stwd_mem_o = Signal(reset_less=True) # activate memory STORE

    def elaborate(self, platform):
        m = Module()
        m.submodules.alu = self.alu
        m.submodules.src_l = src_l = SRLatch(sync=False)
        m.submodules.opc_l = opc_l = SRLatch(sync=False)
        m.submodules.adr_l = adr_l = SRLatch(sync=False)
        m.submodules.req_l = req_l = SRLatch(sync=False)
        m.submodules.sto_l = sto_l = SRLatch(sync=False)

        # shadow/go_die
        reset_w = Signal(reset_less=True)
        reset_a = Signal(reset_less=True)
        reset_s = Signal(reset_less=True)
        reset_r = Signal(reset_less=True)
        m.d.comb += reset_b.eq(self.go_st_ | self.go_wr_i | self.go_die_i)
        m.d.comb += reset_w.eq(self.go_wr_i | self.go_die_i)
        m.d.comb += reset_s.eq(self.go_st_i | self.go_die_i)
        m.d.comb += reset_r.eq(self.go_rd_i | self.go_die_i)
        # this one is slightly different, issue_alu_i selects go_wr_i)
        a_sel = Mux(self.isalu_i, self.go_wr_i, self.go_ad_i )
        m.d.comb += reset_a.eq(a_sel| self.go_die_i)

        # opcode decode
        op_alu = Signal(reset_less=True)
        op_is_ld = Signal(reset_less=True)
        op_is_st = Signal(reset_less=True)
        op_is_imm = Signal(reset_less=True)
        m.d.comb += op_alu.eq(self.oper_i[0])
        m.d.comb += op_is_imm.eq(self.oper_i[1])
        m.d.comb += op_is_ld.eq(self.oper_i[2])
        m.d.comb += op_is_st.eq(self.oper_i[3])
        m.d.comb += self.load_mem_o.eq(op_is_ld & self.go_ad_i)
        m.d.comb += self.stwd_mem_o.eq(op_is_st & self.go_st_i)

        # select immediate or src2 reg to add
        src2_or_imm = Signal(self.rwid, reset_less=True)
        src_sel = Signal(reset_less=True)

        # issue can be either issue_i or issue_alu_i (isalu_i)
        issue_i = Signal(reset_less=True)
        m.d.comb += issue_i.eq(self.issue_i | self.isalu_i)

        # Ripple-down the latches, each one set cancels the previous.
        # NOTE: use sync to stop combinatorial loops.

        # opcode latch - inverted so that busy resets to 0
        m.d.sync += opc_l.s.eq(issue_i) # XXX NOTE: INVERTED FROM book!
        m.d.sync += opc_l.r.eq(reset_b) # XXX NOTE: INVERTED FROM book!

        # src operand latch
        m.d.sync += src_l.s.eq(issue_i)
        m.d.sync += src_l.r.eq(reset_r)

        # addr latch 
        m.d.sync += req_l.s.eq(self.go_rd_i)
        m.d.sync += req_l.r.eq(reset_a)

        # dest operand latch 
        m.d.sync += req_l.s.eq(self.go_ad_i)
        m.d.sync += req_l.r.eq(reset_w)

        # dest operand latch 
        m.d.sync += req_l.s.eq(self.go_ad_i)
        m.d.sync += req_l.r.eq(reset_s)

        # outputs
        m.d.comb += self.busy_o.eq(opc_l.q) # busy out
        m.d.comb += self.rd_rel_o.eq(src_l.q & opc_l.q) # src1/src2 req rel

        # the counter is just for demo purposes, to get the ALUs of different
        # types to take arbitrary completion times
        with m.If(opc_l.qn):
            m.d.sync += self.counter.eq(0)
        with m.If(req_l.qn & opc_l.q & (self.counter == 0)):
            with m.If(self.oper_i == 2): # MUL, to take 5 instructions
                m.d.sync += self.counter.eq(5)
            with m.Elif(self.oper_i == 3): # SHIFT to take 7
                m.d.sync += self.counter.eq(7)
            with m.Else(): # ADD/SUB to take 2
                m.d.sync += self.counter.eq(2)
        with m.If(self.counter > 1):
            m.d.sync += self.counter.eq(self.counter - 1)
        with m.If(self.counter == 1):
            # write req release out.  waits until shadow is dropped.
            m.d.comb += self.req_rel_o.eq(req_l.q & opc_l.q & self.shadown_i)

        # select immediate if opcode says so.  however also change the latch
        # to trigger *from* the opcode latch instead.
        m.d.comb += src_sel.eq(Mux(op_is_imm, opc_l.q, src_l.q))
        m.d.comb += src2_or_imm.eq(Mux(op_is_imm, self.imm_i, self.src2_i)

        # create a latch/register for src1/src2 (include immediate select)
        latchregister(m, self.src1_i, self.alu.a, src_l.q)
        latchregister(m, src2_or_imm, self.alu.b, src_sel)

        # create a latch/register for the operand
        latchregister(m, self.oper_i, self.alu.op, self.issue_i)

        # and one for the output from the ALU
        data_r = Signal(self.rwid, reset_less=True) # Dest register
        latchregister(m, self.alu.o, data_r, req_l.q)

        with m.If(self.go_wr_i):
            m.d.comb += self.data_o.eq(data_r)

        return m

def scoreboard_sim(dut):
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
    yield dut.go_read_i.eq(1)
    yield
    yield dut.go_read_i.eq(0)
    yield
    yield dut.go_write_i.eq(1)
    yield
    yield dut.go_write_i.eq(0)
    yield

def test_scoreboard():
    dut = Scoreboard(32, 8)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_scoreboard.il", "w") as f:
        f.write(vl)

    run_simulation(dut, scoreboard_sim(dut), vcd_name='test_scoreboard.vcd')

if __name__ == '__main__':
    test_scoreboard()
