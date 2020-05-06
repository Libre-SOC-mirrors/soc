""" LOAD / STORE Computation Unit.

    This module covers POWER9-compliant Load and Store operations,
    with selection on each between immediate and indexed mode as
    options for the calculation of the Effective Address (EA),
    and also "update" mode which optionally stores that EA into
    an additional register.

    ----
    Note: it took 15 attempts over several weeks to redraw the diagram
    needed to capture this FSM properly.  To understand it fully, please
    take the time to review the links, video, and diagram.
    ----

    Stores are activated when Go_Store is enabled, and use a sync'd "ADD" to
    compute the "Effective Address", and, when ready the operand (src3_i)
    is stored in the computed address (passed through to the PortInterface)

    Loads are activated when Go_Write[0] is enabled.  The EA is computed,
    and (as long as there was no exception) the data comes out (at any
    time from the PortInterface), and is captured by the LDCompSTUnit.

    Both LD and ST may request that the address be computed from summing
    operand1 (src[0]) with operand2 (src[1]) *or* by summing operand1 with
    the immediate (from the opcode).

    Both LD and ST may also request "update" mode (op_is_update) which
    activates the use of Go_Write[1] to control storage of the EA into
    a *second* operand in the register file.

    Thus this module has *TWO* write-requests to the register file and
    *THREE* read-requests to the register file (not all at the same time!)
    The regfile port usage is:

    * LD-imm         1R1W
    * LD-imm-update  1R2W
    * LD-idx         2R1W
    * LD-idx-update  2R2W

    * ST-imm         2R
    * ST-imm-update  2R1W
    * ST-idx         3R
    * ST-idx-update  3R1W

    It's a multi-level Finite State Machine that (unfortunately) nmigen.FSM
    is not suited to (nmigen.FSM is clock-driven, and some aspects of
    the nested FSMs below are *combinatorial*).

    * One FSM covers Operand collection and communication address-side
      with the LD/ST PortInterface.  its role ends when "RD_DONE" is asserted

    * A second FSM activates to cover LD.  it activates if op_is_ld is true

    * A third FSM activates to cover ST.  it activates if op_is_st is true

    * The "overall" (fourth) FSM coordinates the progression and completion
      of the three other FSMs, firing "WR_RESET" which switches off "busy"

    Full diagram:
    https://libre-soc.org/3d_gpu/ld_st_comp_unit.jpg

    Links including to walk-through videos:
    * https://libre-soc.org/3d_gpu/architecture/6600scoreboard/

    Related Bugreports:
    * https://bugs.libre-soc.org/show_bug.cgi?id=302

    Terminology:

    * EA - Effective Address
    * LD - Load
    * ST - Store
"""

from nmigen.compat.sim import run_simulation
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Mux, Cat, Elaboratable, Array, Repl
from nmigen.hdl.rec import Record, Layout

from nmutil.latch import SRLatch, latchregister

from soc.experiment.compalu_multi import go_record
from soc.experiment.l0_cache import PortInterface
from soc.experiment.testmem import TestMemory
from soc.decoder.power_enums import InternalOp

from soc.decoder.power_enums import InternalOp, Function


class CompLDSTOpSubset(Record):
    """CompLDSTOpSubset

    a copy of the relevant subset information from Decode2Execute1Type
    needed for LD/ST operations.  use with eq_from_execute1 (below) to
    grab subsets.
    """
    def __init__(self, name=None):
        layout = (('insn_type', InternalOp),
                  ('imm_data', Layout((("imm", 64), ("imm_ok", 1)))),
                  ('is_32bit', 1),
                  ('is_signed', 1),
                  ('data_len', 4), # TODO: should be in separate CompLDSTSubset
                  ('byte_reverse', 1),
                  ('sign_extend', 1),
                  ('update', 1))

        Record.__init__(self, Layout(layout), name=name)

        # grrr.  Record does not have kwargs
        self.insn_type.reset_less = True
        self.is_32bit.reset_less = True
        self.is_signed.reset_less = True
        self.data_len.reset_less = True
        self.byte_reverse.reset_less = True
        self.sign_extend.reset_less = True
        self.update.reset_less = True

    def eq_from_execute1(self, other):
        """ use this to copy in from Decode2Execute1Type
        """
        res = []
        for fname, sig in self.fields.items():
            eqfrom = other.fields[fname]
            res.append(sig.eq(eqfrom))
        return res

    def ports(self):
        return [self.insn_type,
                self.is_32bit,
                self.is_signed,
                self.data_len,
                self.byte_reverse,
                self.sign_extend,
                self.update,
        ]


class LDSTCompUnit(Elaboratable):
    """LOAD / STORE Computation Unit

    Inputs
    ------

    * :pi:     a PortInterface to the memory subsystem (read-write capable)
    * :rwid:   register width
    * :awid:   address width

    Data inputs
    -----------
    * :src_i:  Source Operands (RA/RB/RC) - managed by rd[0-3] go/req

    Data (outputs)
    --------------
    * :data_o:     Dest out (LD)          - managed by wr[0] go/req
    * :addr_o:     Address out (LD or ST) - managed by wr[1] go/req
    * :addr_exc_o: Address/Data Exception occurred.  LD/ST must terminate

    TODO: make addr_exc_o a data-type rather than a single-bit signal
          (see bug #302)

    Control Signals (In)
    --------------------

    * :oper_i:     operation being carried out (POWER9 decode LD/ST subset)
    * :issue_i:    LD/ST is being "issued".
    * :shadown_i:  Inverted-shadow is being held (stops STORE *and* WRITE)
    * :go_rd_i:    read is being actioned (latches in src regs)
    * :go_wr_i:    write mode (exactly like ALU CompUnit)
    * :go_ad_i:    address is being actioned (triggers actual mem LD)
    * :go_st_i:    store is being actioned (triggers actual mem STORE)
    * :go_die_i:   resets the unit back to "wait for issue"

    Control Signals (Out)
    ---------------------

    * :busy_o:      function unit is busy
    * :rd_rel_o:    request src1/src2
    * :adr_rel_o:   request address (from mem)
    * :sto_rel_o:   request store (to mem)
    * :req_rel_o:   request write (result)
    * :load_mem_o:  activate memory LOAD
    * :stwd_mem_o:  activate memory STORE

    Note: load_mem_o, stwd_mem_o and req_rel_o MUST all be acknowledged
    in a single cycle and the CompUnit set back to doing another op.
    This means deasserting go_st_i, go_ad_i or go_wr_i as appropriate
    depending on whether the operation is a ST or LD.
    """

    def __init__(self, pi, rwid=64, awid=48, debugtest=False):
        self.rwid = rwid
        self.awid = awid
        self.pi = pi
        self.debugtest = debugtest

        # POWER-compliant LD/ST has index and update: *fixed* number of ports
        self.n_src = n_src = 3   # RA, RB, RT/RS
        self.n_dst = n_dst = 2 # RA, RT/RS

        # set up array of src and dest signals
        src = []
        for i in range(n_src):
            j = i + 1 # name numbering to match src1/src2
            src.append(Signal(rwid, name="src%d_i" % j, reset_less=True))

        dst = []
        for i in range(n_dst):
            j = i + 1 # name numbering to match dest1/2...
            dst.append(Signal(rwid, name="dest%d_i" % j, reset_less=True))

        # control (dual in/out)
        self.rd = go_record(n_src, name="rd") # read in, req out
        self.wr = go_record(n_dst, name="wr") # write in, req out
        self.ad = go_record(1, name="ad") # address go in, req out
        self.st = go_record(1, name="st") # store go in, req out

        self.go_rd_i = self.rd.go # temporary naming
        self.go_wr_i = self.wr.go # temporary naming
        self.rd_rel_o = self.rd.rel # temporary naming
        self.req_rel_o = self.wr.rel # temporary naming

        self.go_ad_i = self.ad.go # temp naming: go address in
        self.go_st_i = self.st.go  # temp naming: go store in

        # control inputs
        self.issue_i = Signal(reset_less=True)  # fn issue in
        self.shadown_i = Signal(reset=1)  # shadow function, defaults to ON
        self.go_die_i = Signal()  # go die (reset)

        # operation / data input
        self.oper_i = CompLDSTOpSubset() # operand
        self.src_i = Array(src)
        self.src1_i = src[0] # oper1 in: RA
        self.src2_i = src[1] # oper2 in: RB
        self.src3_i = src[2] # oper2 in: RC (RS)

        # outputs
        self.busy_o = Signal(reset_less=True)       # fn busy out
        self.done_o = Signal(reset_less=True)  # final release signal
        # TODO (see bug #302)
        self.addr_exc_o = Signal(reset_less=True)   # address exception
        self.dest = Array(dst)
        self.data_o = dst[0]  # Dest1 out: RT
        self.addr_o = dst[1]  # Address out (LD or ST) - Update => RA

        self.adr_rel_o = self.ad.rel  # request address (from mem)
        self.sto_rel_o = self.st.rel  # request store (to mem)

        self.ld_o = Signal(reset_less=True)  # operation is a LD
        self.st_o = Signal(reset_less=True)  # operation is a ST

        # hmm... are these necessary?
        self.load_mem_o = Signal(reset_less=True)  # activate memory LOAD
        self.stwd_mem_o = Signal(reset_less=True)  # activate memory STORE

    def elaborate(self, platform):
        m = Module()

        # temp/convenience
        comb = m.d.comb
        sync = m.d.sync
        issue_i = self.issue_i

        #####################
        # latches for the FSM.
        m.submodules.opc_l = opc_l = SRLatch(sync=False, name="opc")
        m.submodules.src_l = src_l = SRLatch(False, self.n_src, name="src")
        m.submodules.alu_l = alu_l = SRLatch(sync=False, name="alu")
        m.submodules.adr_l = adr_l = SRLatch(sync=False, name="adr")
        m.submodules.lod_l = lod_l = SRLatch(sync=False, name="lod")
        m.submodules.sto_l = sto_l = SRLatch(sync=False, name="sto")
        m.submodules.wri_l = wri_l = SRLatch(sync=False, name="wri")
        m.submodules.upd_l = upd_l = SRLatch(sync=False, name="upd")
        m.submodules.rst_l = rst_l = SRLatch(sync=False, name="rst")

        ####################
        # signals

        # opcode decode
        op_is_ld = Signal(reset_less=True)
        op_is_st = Signal(reset_less=True)

        # ALU/LD data output control
        alu_valid = Signal(reset_less=True) # ALU operands are valid
        alu_ok = Signal(reset_less=True)    # ALU out ok (1 clock delay valid)
        addr_ok = Signal(reset_less=True)   # addr ok (from PortInterface)
        ld_ok = Signal(reset_less=True)     # LD out ok from PortInterface
        wr_any = Signal(reset_less=True)    # any write (incl. store)
        rd_done = Signal(reset_less=True)   # all *necessary* operands read
        wr_reset = Signal(reset_less=True)  # final reset condition

        # LD and ALU out
        alu_o = Signal(self.rwid, reset_less=True)
        ld_o = Signal(self.rwid, reset_less=True)

        # select immediate or src2 reg to add
        src2_or_imm = Signal(self.rwid, reset_less=True)
        src_sel = Signal(reset_less=True)

        ##############################
        # reset conditions for latches

        # temporaries (also convenient when debugging)
        reset_o = Signal(reset_less=True)             # reset opcode
        reset_w = Signal(reset_less=True)             # reset write
        reset_u = Signal(reset_less=True)             # reset update
        reset_a = Signal(reset_less=True)             # reset adr latch
        reset_i = Signal(reset_less=True)             # issue|die (use a lot)
        reset_r = Signal(self.n_src, reset_less=True) # reset src
        reset_s = Signal(reset_less=True)             # reset store

        comb += reset_i.eq(issue_i | self.go_die_i)       # various
        comb += reset_o.eq(wr_reset | self.go_die_i)      # opcode reset
        comb += reset_w.eq(self.wr.go[0] | self.go_die_i) # write reg 1
        comb += reset_u.eq(self.wr.go[1] | self.go_die_i) # update (reg 2)
        comb += reset_s.eq(self.go_st_i | self.go_die_i)  # store reset
        comb += reset_r.eq(self.rd.go | Repl(self.go_die_i, self.n_src))
        comb += reset_a.eq(self.go_ad_i | self.go_die_i)

        ##########################
        # FSM implemented through sequence of latches.  approximately this:
        # - opc_l       : opcode
        #    - src_l[0] : operands
        #    - src_l[1]
        #       - alu_l : looks after add of src1/2/imm (EA)
        #       - adr_l : waits for add (EA)
        #       - upd_l : waits for adr and Regfile (port 2)
        #    - src_l[2] : ST
        # - lod_l       : waits for adr (EA) and for LD Data
        # - wri_l       : waits for LD Data and Regfile (port 1)
        # - st_l        : waits for alu and operand2
        # - rst_l       : waits for all FSM paths to converge.
        # NOTE: use sync to stop combinatorial loops.

        # opcode latch - inverted so that busy resets to 0
        # note this MUST be sync so as to avoid a combinatorial loop
        # between busy_o and issue_i on the reset latch (rst_l)
        sync += opc_l.s.eq(issue_i)  # XXX NOTE: INVERTED FROM book!
        sync += opc_l.r.eq(reset_o)  # XXX NOTE: INVERTED FROM book!

        # src operand latch
        sync += src_l.s.eq(Repl(issue_i, self.n_src))
        sync += src_l.r.eq(reset_r)

        # alu latch
        comb += alu_l.s.eq(alu_ok)
        comb += alu_l.r.eq(reset_i)

        # addr latch
        comb += adr_l.s.eq(reset_a)
        comb += adr_l.r.eq(alu_ok)

        # ld latch
        comb += lod_l.s.eq(reset_i)
        comb += lod_l.r.eq(ld_ok)

        # dest operand latch
        sync += wri_l.s.eq(issue_i)
        sync += wri_l.r.eq(reset_w)

        # update-mode operand latch (EA written to reg 2)
        sync += upd_l.s.eq(alu_ok)
        sync += upd_l.r.eq(reset_u)

        # store latch
        sync += sto_l.s.eq(addr_ok & op_is_st)
        sync += sto_l.r.eq(reset_s)

        # reset latch
        comb += rst_l.s.eq(addr_ok) # start when address is ready
        comb += rst_l.r.eq(issue_i)

        # create a latch/register for the operand
        oper_r = CompLDSTOpSubset()  # Dest register
        latchregister(m, self.oper_i, oper_r, self.issue_i, name="oper_r")

        # and for LD
        ldd_r = Signal(self.rwid, reset_less=True)  # Dest register
        latchregister(m, ld_o, ldd_r, lod_l.q, "ldo_r")

        # and for each input from the incoming src operands
        srl = []
        for i in range(self.n_src):
            name = "src_r%d" % i
            src_r = Signal(self.rwid, name=name, reset_less=True)
            latchregister(m, self.src_i[i], src_r, src_l.q[i], name)
            srl.append(src_r)

        # and one for the output from the ADD (for the EA)
        addr_r = Signal(self.rwid, reset_less=True)  # Effective Address Latch
        latchregister(m, alu_o, addr_r, alu_l.q, "ea_r")

        # select either immediate or src2 if opcode says so
        op_is_imm = oper_r.imm_data.imm_ok
        src2_or_imm = Signal(self.rwid, reset_less=True)
        m.d.comb += src2_or_imm.eq(Mux(op_is_imm, oper_r.imm_data.imm, srl[0]))

        # now do the ALU addr add: one cycle, and say "ready" (next cycle, too)
        sync += alu_o.eq(src_r[0] + src2_or_imm) # actual EA
        sync += alu_ok.eq(alu_valid)             # keep ack in sync with EA

        # outputs: busy and release signals
        busy_o = self.busy_o
        comb += self.busy_o.eq(opc_l.q)  # busy out
        comb += self.rd.rel.eq(src_l.q & busy_o)  # src1/src2 req rel
        comb += self.sto_rel_o.eq(sto_l.q & busy_o & self.shadown_i & op_is_st)

        # decode bits of operand (latched)
        comb += op_is_st.eq(oper_r.insn_type == InternalOp.OP_STORE) # ST
        comb += op_is_ld.eq(oper_r.insn_type == InternalOp.OP_LOAD)  # LD
        op_is_update = oper_r.update                                 # UPDATE
        comb += self.load_mem_o.eq(op_is_ld & self.go_ad_i)
        comb += self.stwd_mem_o.eq(op_is_st & self.go_st_i)
        comb += self.ld_o.eq(op_is_ld)
        comb += self.st_o.eq(op_is_st)

        ############################
        # Control Signal calculation

        # 1st operand read-request is simple: always need it
        comb += self.rd.rel[0].eq(src_l.q[0] & busy_o)

        # 2nd operand only needed when immediate is not active
        comb += self.rd.rel[1].eq(src_l.q[1] & busy_o & ~op_is_imm)

        # alu input valid when 1st and 2nd ops done (or imm not active)
        comb += alu_valid.eq(busy_o & ~(self.rd.rel[0] | self.rd.rel[1]))

        # 3rd operand only needed when operation is a store
        comb += self.rd.rel[2].eq(src_l.q[2] & busy_o & op_is_st)

        # all reads done when alu is valid and 3rd operand needed
        comb += rd_done.eq(alu_valid & ~self.rd.rel[2])

        # address release only if addr ready, but Port must be idle
        comb += self.adr_rel_o.eq(adr_l.q & busy_o & ~self.pi.busy_o)

        # store release when st ready *and* all operands read (and no shadow)
        comb += self.st.rel.eq(sto_l.q & busy_o & rd_done & op_is_st &
                               self.shadown_i)

        # request write of LD result.  waits until shadow is dropped.
        comb += self.wr.rel[0].eq(wri_l.q & busy_o & lod_l.qn & op_is_ld &
                                  self.shadown_i)

        # request write of EA result only in update mode
        comb += self.wr.rel[0].eq(upd_l.q & busy_o & op_is_update &
                                  self.shadown_i)

        # provide "done" signal: select req_rel for non-LD/ST, adr_rel for LD/ST
        comb += wr_any.eq(self.st.go | self.wr.go[0] | self.wr.go[1])
        comb += wr_reset.eq(rst_l.q & busy_o & self.shadown_i & wr_any &
                    ~(self.st.rel | self.wr.rel[0] | self.wr.rel[1]) & lod_l.qn)
        comb += self.done_o.eq(wr_reset)

        ######################
        # Data/Address outputs

        # put the LD-output register directly onto the output bus on a go_write
        with m.If(self.wr.go[0]):
            comb += self.data_o.eq(ldd_r)

        # "update" mode, put address out on 2nd go-write
        with m.If(op_is_update & self.wr.go[1]):
            comb += self.addr_o.eq(addr_r)

        ###########################
        # PortInterface connections
        pi = self.pi

        # connect to LD/ST PortInterface.
        comb += pi.is_ld_i.eq(op_is_ld)  # decoded-LD
        comb += pi.is_st_i.eq(op_is_st)  # decoded-ST
        comb += pi.op.eq(self.oper_i)    # op details (not all needed)
        # address
        comb += pi.addr.data.eq(addr_r)           # EA from adder
        comb += pi.addr.ok.eq(self.ad.go)         # "go do address stuff"
        comb += self.addr_exc_o.eq(pi.addr_exc_o) # exception occurred
        comb += addr_ok.eq(self.pi.addr_ok_o)     # no exc, address fine
        # ld - ld gets latched in via lod_l
        comb += ld_o.eq(pi.ld.data)  # ld data goes into ld reg (above)
        comb += ld_ok.eq(pi.ld.ok) # ld.ok *closes* (freezes) ld data
        # store - data goes in based on go_st
        comb += pi.st.data.eq(srl[2]) # 3rd operand latch
        comb += pi.st.ok.eq(self.st.go)  # go store signals st data valid

        return m

    def __iter__(self):
        yield self.rd.go
        yield self.go_ad_i
        yield self.wr.go
        yield self.go_st_i
        yield self.issue_i
        yield self.shadown_i
        yield self.go_die_i
        yield from self.oper_i.ports()
        yield from self.src_i
        yield self.busy_o
        yield self.rd.rel
        yield self.adr_rel_o
        yield self.sto_rel_o
        yield self.wr.rel
        yield self.data_o
        yield self.addr_o
        yield self.load_mem_o
        yield self.stwd_mem_o

    def ports(self):
        return list(self)


def wait_for(sig):
    v = (yield sig)
    print("wait for", sig, v)
    while True:
        yield
        v = (yield sig)
        print(v)
        if v:
            break


def store(dut, src1, src2, imm, imm_ok=True):
    yield dut.oper_i.insn_type.eq(InternalOp.OP_STORE)
    yield dut.src1_i.eq(src1)
    yield dut.src2_i.eq(src2)
    yield dut.oper_i.imm_data.imm.eq(imm)
    yield dut.oper_i.imm_data.imm_ok.eq(imm_ok)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.rd.go.eq(0b11)
    yield from wait_for(dut.rd.rel)
    yield dut.rd.go.eq(0)
    yield from wait_for(dut.adr_rel_o)
    yield dut.go_st_i.eq(1)
    yield from wait_for(dut.sto_rel_o)
    wait_for(dut.stwd_mem_o)
    yield dut.go_st_i.eq(0)
    yield


def load(dut, src1, src2, imm, imm_ok=True):
    yield dut.oper_i.insn_type.eq(InternalOp.OP_LOAD)
    yield dut.src1_i.eq(src1)
    yield dut.src2_i.eq(src2)
    yield dut.oper_i.imm_data.imm.eq(imm)
    yield dut.oper_i.imm_data.imm_ok.eq(imm_ok)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.rd.go.eq(0b11)
    yield from wait_for(dut.rd.rel)
    yield dut.rd.go.eq(0)
    yield from wait_for(dut.adr_rel_o)
    yield dut.go_ad_i.eq(1)
    yield from wait_for(dut.busy_o)
    yield
    data = (yield dut.data_o)
    yield dut.go_ad_i.eq(0)
    # wait_for(dut.stwd_mem_o)
    return data


def add(dut, src1, src2, imm, imm_ok=False):
    yield dut.oper_i.insn_type.eq(InternalOp.OP_ADD)
    yield dut.src1_i.eq(src1)
    yield dut.src2_i.eq(src2)
    yield dut.oper_i.imm_data.imm.eq(imm)
    yield dut.oper_i.imm_data.imm_ok.eq(imm_ok)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.rd.go.eq(1)
    yield from wait_for(dut.rd.rel)
    yield dut.rd.go.eq(0)
    yield from wait_for(dut.wr.rel)
    yield dut.wr.go.eq(1)
    yield from wait_for(dut.busy_o)
    yield
    data = (yield dut.data_o)
    yield dut.wr.go.eq(0)
    yield
    # wait_for(dut.stwd_mem_o)
    return data


def scoreboard_sim(dut):
    # two STs (different addresses)
    yield from store(dut, 4, 3, 2)
    yield from store(dut, 2, 9, 2)
    yield
    # two LDs (deliberately LD from the 1st address then 2nd)
    data = yield from load(dut, 4, 0, 2)
    assert data == 0x0003
    data = yield from load(dut, 2, 0, 2)
    assert data == 0x0009
    yield

    # now do an add
    data = yield from add(dut, 4, 3, 0xfeed)
    assert data == 0x7

    # and an add-immediate
    data = yield from add(dut, 4, 0xdeef, 2, imm_ok=True)
    assert data == 0x6


class TestLDSTCompUnit(LDSTCompUnit):

    def __init__(self, rwid):
        from soc.experiment.l0_cache import TstL0CacheBuffer
        self.l0 = l0 = TstL0CacheBuffer()
        pi = l0.l0.dports[0].pi
        LDSTCompUnit.__init__(self, pi, rwid, 4)

    def elaborate(self, platform):
        m = LDSTCompUnit.elaborate(self, platform)
        m.submodules.l0 = self.l0
        return m


def test_scoreboard():

    dut = TestLDSTCompUnit(16)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_ldst_comp.il", "w") as f:
        f.write(vl)

    run_simulation(dut, scoreboard_sim(dut), vcd_name='test_ldst_comp.vcd')


if __name__ == '__main__':
    test_scoreboard()
