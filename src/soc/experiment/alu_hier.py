"""*Experimental* ALU: based on nmigen alu_hier.py, includes branch-compare ALU

This ALU is *deliberately* designed to add in (unnecessary) delays into
different operations so as to be able to test the 6600-style matrices
and the CompUnits.  Countdown timers wait for (defined) periods before
indicating that the output is valid

A "real" integer ALU would place the answers onto the output bus after
only one cycle (sync)
"""

from nmigen import Elaboratable, Signal, Module, Const, Mux
from nmigen.hdl.rec import Record, Layout
from nmigen.cli import main
from nmigen.cli import verilog, rtlil
from nmigen.compat.sim import run_simulation
from nmutil.extend import exts
from nmutil.gtkw import write_gtkw

# NOTE: to use cxxsim, export NMIGEN_SIM_MODE=cxxsim from the shell
# Also, check out the cxxsim nmigen branch, and latest yosys from git
from nmutil.sim_tmp_alternative import (Simulator, nmigen_sim_top_module,
                                        is_engine_pysim)

from openpower.decoder.decode2execute1 import Data
from openpower.decoder.power_enums import MicrOp, Function, CryIn

from soc.fu.alu.alu_input_record import CompALUOpSubset
from soc.fu.cr.cr_input_record import CompCROpSubset

from soc.fu.pipe_data import FUBaseData
from soc.fu.alu.pipe_data import CommonPipeSpec
from soc.fu.compunits.compunits import FunctionUnitBaseSingle

import operator


class Adder(Elaboratable):
    def __init__(self, width):
        self.invert_in = Signal()
        self.a = Signal(width)
        self.b = Signal(width)
        self.o = Signal(width, name="add_o")

    def elaborate(self, platform):
        m = Module()
        with m.If(self.invert_in):
            m.d.comb += self.o.eq((~self.a) + self.b)
        with m.Else():
            m.d.comb += self.o.eq(self.a + self.b)
        return m


class Subtractor(Elaboratable):
    def __init__(self, width):
        self.a = Signal(width)
        self.b = Signal(width)
        self.o = Signal(width, name="sub_o")

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(self.a - self.b)
        return m


class Multiplier(Elaboratable):
    def __init__(self, width):
        self.a = Signal(width)
        self.b = Signal(width)
        self.o = Signal(width, name="mul_o")

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(self.a * self.b)
        return m


class Shifter(Elaboratable):
    def __init__(self, width):
        self.width = width
        self.a = Signal(width)
        self.b = Signal(width)
        self.o = Signal(width, name="shf_o")

    def elaborate(self, platform):
        m = Module()
        btrunc = Signal(self.width)
        m.d.comb += btrunc.eq(self.b & Const((1 << self.width)-1))
        m.d.comb += self.o.eq(self.a >> btrunc)
        return m


class SignExtend(Elaboratable):
    def __init__(self, width):
        self.width = width
        self.a = Signal(width)
        self.o = Signal(width, name="exts_o")

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(exts(self.a, 8, self.width))
        return m


class Dummy:
    pass


class DummyALU(Elaboratable):
    def __init__(self, width):
        self.p = Dummy()  # make look like nmutil pipeline API
        self.p.i_data = Dummy()
        self.p.i_data.ctx = Dummy()
        self.n = Dummy()  # make look like nmutil pipeline API
        self.n.o_data = Dummy()
        self.p.i_valid = Signal()
        self.p.o_ready = Signal()
        self.n.i_ready = Signal()
        self.n.o_valid = Signal()
        self.counter = Signal(4)
        self.op = CompCROpSubset()
        i = []
        i.append(Signal(width, name="i1"))
        i.append(Signal(width, name="i2"))
        i.append(Signal(width, name="i3"))
        self.i = i
        self.a, self.b, self.c = i[0], i[1], i[2]
        self.out = tuple([Signal(width, name="alu_o")])
        self.o = self.out[0]
        self.width = width
        # more "look like nmutil pipeline API"
        self.p.i_data.ctx.op = self.op
        self.p.i_data.a = self.a
        self.p.i_data.b = self.b
        self.p.i_data.c = self.c
        self.n.o_data.o = self.o

    def elaborate(self, platform):
        m = Module()

        go_now = Signal(reset_less=True)  # testing no-delay ALU

        with m.If(self.p.i_valid):
            # input is valid. next check, if we already said "ready" or not
            with m.If(~self.p.o_ready):
                # we didn't say "ready" yet, so say so and initialise
                m.d.sync += self.p.o_ready.eq(1)

                m.d.sync += self.o.eq(self.a)
                m.d.comb += go_now.eq(1)
                m.d.sync += self.counter.eq(1)

        with m.Else():
            # input says no longer valid, so drop ready as well.
            # a "proper" ALU would have had to sync in the opcode and a/b ops
            m.d.sync += self.p.o_ready.eq(0)

        # ok so the counter's running: when it gets to 1, fire the output
        with m.If((self.counter == 1) | go_now):
            # set the output as valid if the recipient is ready for it
            m.d.sync += self.n.o_valid.eq(1)
        with m.If(self.n.i_ready & self.n.o_valid):
            m.d.sync += self.n.o_valid.eq(0)
            # recipient said it was ready: reset back to known-good.
            m.d.sync += self.counter.eq(0)  # reset the counter
            m.d.sync += self.o.eq(0)  # clear the output for tidiness sake

        # countdown to 1 (transition from 1 to 0 only on acknowledgement)
        with m.If(self.counter > 1):
            m.d.sync += self.counter.eq(self.counter - 1)

        return m

    def __iter__(self):
        yield from self.op.ports()
        yield self.a
        yield self.b
        yield self.c
        yield self.o

    def ports(self):
        return list(self)

#####################
# converting even this dummy ALU over to the FunctionUnit RegSpecs API
# which, errr, note that the regspecs are totally ignored below, but
# at least the widths are all 64-bit so it's okay.
#####################

# input (and output) for logical initial stage (common input)


class ALUInputData(FUBaseData):
    regspec = [('INT', 'a', '0:63'),  # RA
               ('INT', 'b', '0:63'),  # RB/immediate
               ]

    def __init__(self, pspec):
        super().__init__(pspec, False)


# output from ALU final stage
class ALUOutputData(FUBaseData):
    regspec = [('INT', 'o', '0:63'),        # RT
               ]

    def __init__(self, pspec):
        super().__init__(pspec, True)


# ALU pipe specification class
class ALUPipeSpec(CommonPipeSpec):
    regspec = (ALUInputData.regspec, ALUOutputData.regspec)
    opsubsetkls = CompALUOpSubset


class ALUFunctionUnit(FunctionUnitBaseSingle):
    # class ALUFunctionUnit(FunctionUnitBaseMulti):
    fnunit = Function.ALU

    def __init__(self, idx):
        super().__init__(ALUPipeSpec, ALU, 1)


class ALU(Elaboratable):
    def __init__(self, width):
        # XXX major temporary hack: attempting to convert
        # ALU over to RegSpecs API, FunctionUnitBaseSingle passes in
        # a regspec here which we can't cope with.  therefore, errr...
        # just throw it away and set the width to 64
        if not isinstance(width, int):
            width = 64
        # TODO, really this should just inherit from ControlBase it would
        # be a lot less messy.
        self.p = Dummy()  # make look like nmutil pipeline API
        self.p.i_data = Dummy()
        self.p.i_data.ctx = Dummy()
        self.n = Dummy()  # make look like nmutil pipeline API
        self.n.o_data = Dummy()
        self.p.i_valid = Signal()
        self.p.o_ready = Signal()
        self.n.i_ready = Signal()
        self.n.o_valid = Signal()
        self.counter = Signal(4)
        self.op = CompALUOpSubset(name="op")
        i = []
        i.append(Signal(width, name="i1"))
        i.append(Signal(width, name="i2"))
        self.i = i
        self.a, self.b = i[0], i[1]
        out = []
        out.append(Data(width, name="alu_o"))
        out.append(Data(width, name="alu_cr"))
        self.out = tuple(out)
        self.o = self.out[0]
        self.cr = self.out[1]
        self.width = width
        # more "look like nmutil ControlBase pipeline API" stuff
        self.p.i_data.ctx.op = self.op
        self.p.i_data.a = self.a
        self.p.i_data.b = self.b
        self.n.o_data.o = self.o
        self.n.o_data.cr = self.cr

    def elaborate(self, platform):
        m = Module()
        add = Adder(self.width)
        mul = Multiplier(self.width)
        shf = Shifter(self.width)
        sub = Subtractor(self.width)
        ext_sign = SignExtend(self.width)

        m.submodules.add = add
        m.submodules.mul = mul
        m.submodules.shf = shf
        m.submodules.sub = sub
        m.submodules.ext_sign = ext_sign

        # really should not activate absolutely all ALU inputs like this
        for mod in [add, mul, shf, sub]:
            m.d.comb += [
                mod.a.eq(self.a),
                mod.b.eq(self.b),
            ]
        # EXTS sign extends the first input
        with m.If(self.op.insn_type == MicrOp.OP_EXTS):
            m.d.comb += ext_sign.a.eq(self.a)
        # EXTSWSLI sign extends the second input
        with m.Elif(self.op.insn_type == MicrOp.OP_EXTSWSLI):
            m.d.comb += ext_sign.a.eq(self.b)

        # pass invert (and carry later)
        m.d.comb += add.invert_in.eq(self.op.invert_in)

        go_now = Signal(reset_less=True)  # testing no-delay ALU

        # ALU sequencer is idle when the count is zero
        alu_idle = Signal(reset_less=True)
        m.d.comb += alu_idle.eq(self.counter == 0)

        # ALU sequencer is done when the count is one
        alu_done = Signal(reset_less=True)
        m.d.comb += alu_done.eq(self.counter == 1)

        # select handshake handling according to ALU type
        with m.If(go_now):
            # with a combinatorial, no-delay ALU, just pass through
            # the handshake signals to the other side
            m.d.comb += self.p.o_ready.eq(self.n.i_ready)
            m.d.comb += self.n.o_valid.eq(self.p.i_valid)
        with m.Else():
            # sequential ALU handshake:
            # o_ready responds to i_valid, but only if the ALU is idle
            m.d.comb += self.p.o_ready.eq(alu_idle)
            # select the internally generated o_valid, above
            m.d.comb += self.n.o_valid.eq(alu_done)

        # hold the ALU result until o_ready is asserted
        alu_r = Signal(self.width)

        # output masks
        # NOP and ILLEGAL don't output anything
        with m.If((self.op.insn_type != MicrOp.OP_NOP) &
                  (self.op.insn_type != MicrOp.OP_ILLEGAL)):
            m.d.comb += self.o.ok.eq(1)
        # CR is output when rc bit is active
        m.d.comb += self.cr.ok.eq(self.op.rc.rc)

        with m.If(alu_idle):
            with m.If(self.p.i_valid):

                # as this is a "fake" pipeline, just grab the output right now
                with m.If(self.op.insn_type == MicrOp.OP_ADD):
                    m.d.sync += alu_r.eq(add.o)
                with m.Elif(self.op.insn_type == MicrOp.OP_MUL_L64):
                    m.d.sync += alu_r.eq(mul.o)
                with m.Elif(self.op.insn_type == MicrOp.OP_SHR):
                    m.d.sync += alu_r.eq(shf.o)
                with m.Elif(self.op.insn_type == MicrOp.OP_EXTS):
                    m.d.sync += alu_r.eq(ext_sign.o)
                with m.Elif(self.op.insn_type == MicrOp.OP_EXTSWSLI):
                    m.d.sync += alu_r.eq(ext_sign.o)
                # SUB is zero-delay, no need to register

                # NOTE: all of these are fake, just something to test

                # MUL, to take 5 instructions
                with m.If(self.op.insn_type == MicrOp.OP_MUL_L64):
                    m.d.sync += self.counter.eq(5)
                # SHIFT to take 1, straight away
                with m.Elif(self.op.insn_type == MicrOp.OP_SHR):
                    m.d.sync += self.counter.eq(1)
                # ADD/SUB to take 3
                with m.Elif(self.op.insn_type == MicrOp.OP_ADD):
                    m.d.sync += self.counter.eq(3)
                # EXTS to take 1
                with m.Elif(self.op.insn_type == MicrOp.OP_EXTS):
                    m.d.sync += self.counter.eq(1)
                # EXTSWSLI to take 1
                with m.Elif(self.op.insn_type == MicrOp.OP_EXTSWSLI):
                    m.d.sync += self.counter.eq(1)
                # others to take no delay
                with m.Else():
                    m.d.comb += go_now.eq(1)

        with m.Elif(~alu_done | self.n.i_ready):
            # decrement the counter while the ALU is neither idle nor finished
            m.d.sync += self.counter.eq(self.counter - 1)

        # choose between zero-delay output, or registered
        with m.If(go_now):
            m.d.comb += self.o.data.eq(sub.o)
        # only present the result at the last computation cycle
        with m.Elif(alu_done):
            m.d.comb += self.o.data.eq(alu_r)

        # determine condition register bits based on the data output value
        with m.If(~self.o.data.any()):
            m.d.comb += self.cr.data.eq(0b001)
        with m.Elif(self.o.data[-1]):
            m.d.comb += self.cr.data.eq(0b010)
        with m.Else():
            m.d.comb += self.cr.data.eq(0b100)

        return m

    def __iter__(self):
        yield from self.op.ports()
        yield self.a
        yield self.b
        yield from self.o.ports()
        yield self.p.i_valid
        yield self.p.o_ready
        yield self.n.o_valid
        yield self.n.i_ready

    def ports(self):
        return list(self)


class BranchOp(Elaboratable):
    def __init__(self, width, op):
        self.a = Signal(width)
        self.b = Signal(width)
        self.o = Signal(width)
        self.op = op

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(Mux(self.op(self.a, self.b), 1, 0))
        return m


class BranchALU(Elaboratable):
    def __init__(self, width):
        self.p = Dummy()  # make look like nmutil pipeline API
        self.p.i_data = Dummy()
        self.p.i_data.ctx = Dummy()
        self.n = Dummy()  # make look like nmutil pipeline API
        self.n.o_data = Dummy()
        self.p.i_valid = Signal()
        self.p.o_ready = Signal()
        self.n.i_ready = Signal()
        self.n.o_valid = Signal()
        self.counter = Signal(4)
        self.op = Signal(2)
        i = []
        i.append(Signal(width, name="i1"))
        i.append(Signal(width, name="i2"))
        self.i = i
        self.a, self.b = i[0], i[1]
        self.out = tuple([Signal(width)])
        self.o = self.out[0]
        self.width = width

    def elaborate(self, platform):
        m = Module()
        bgt = BranchOp(self.width, operator.gt)
        blt = BranchOp(self.width, operator.lt)
        beq = BranchOp(self.width, operator.eq)
        bne = BranchOp(self.width, operator.ne)

        m.submodules.bgt = bgt
        m.submodules.blt = blt
        m.submodules.beq = beq
        m.submodules.bne = bne
        for mod in [bgt, blt, beq, bne]:
            m.d.comb += [
                mod.a.eq(self.a),
                mod.b.eq(self.b),
            ]

        go_now = Signal(reset_less=True)  # testing no-delay ALU
        with m.If(self.p.i_valid):
            # input is valid. next check, if we already said "ready" or not
            with m.If(~self.p.o_ready):
                # we didn't say "ready" yet, so say so and initialise
                m.d.sync += self.p.o_ready.eq(1)

                # as this is a "fake" pipeline, just grab the output right now
                with m.Switch(self.op):
                    for i, mod in enumerate([bgt, blt, beq, bne]):
                        with m.Case(i):
                            m.d.sync += self.o.eq(mod.o)
                # branch to take 5 cycles (fake)
                m.d.sync += self.counter.eq(5)
                #m.d.comb += go_now.eq(1)
        with m.Else():
            # input says no longer valid, so drop ready as well.
            # a "proper" ALU would have had to sync in the opcode and a/b ops
            m.d.sync += self.p.o_ready.eq(0)

        # ok so the counter's running: when it gets to 1, fire the output
        with m.If((self.counter == 1) | go_now):
            # set the output as valid if the recipient is ready for it
            m.d.sync += self.n.o_valid.eq(1)
        with m.If(self.n.i_ready & self.n.o_valid):
            m.d.sync += self.n.o_valid.eq(0)
            # recipient said it was ready: reset back to known-good.
            m.d.sync += self.counter.eq(0)  # reset the counter
            m.d.sync += self.o.eq(0)  # clear the output for tidiness sake

        # countdown to 1 (transition from 1 to 0 only on acknowledgement)
        with m.If(self.counter > 1):
            m.d.sync += self.counter.eq(self.counter - 1)

        return m

    def __iter__(self):
        yield self.op
        yield self.a
        yield self.b
        yield self.o

    def ports(self):
        return list(self)


def run_op(dut, a, b, op, inv_a=0):
    yield dut.a.eq(a)
    yield dut.b.eq(b)
    yield dut.op.insn_type.eq(op)
    yield dut.op.invert_in.eq(inv_a)
    yield dut.n.i_ready.eq(0)
    yield dut.p.i_valid.eq(1)
    yield dut.n.i_ready.eq(1)
    yield

    # wait for the ALU to accept our input data
    while not (yield dut.p.o_ready):
        yield

    yield dut.p.i_valid.eq(0)
    yield dut.a.eq(0)
    yield dut.b.eq(0)
    yield dut.op.insn_type.eq(0)
    yield dut.op.invert_in.eq(0)

    # wait for the ALU to present the output data
    while not (yield dut.n.o_valid):
        yield

    # latch the result and lower read_i
    result = yield dut.o.data
    yield dut.n.i_ready.eq(0)

    return result


def alu_sim(dut):
    result = yield from run_op(dut, 5, 3, MicrOp.OP_ADD)
    print("alu_sim add", result)
    assert (result == 8)

    result = yield from run_op(dut, 2, 3, MicrOp.OP_MUL_L64)
    print("alu_sim mul", result)
    assert (result == 6)

    result = yield from run_op(dut, 5, 3, MicrOp.OP_ADD, inv_a=1)
    print("alu_sim add-inv", result)
    assert (result == 65533)

    # test zero-delay ALU
    # don't have OP_SUB, so use any other
    result = yield from run_op(dut, 5, 3, MicrOp.OP_CMP)
    print("alu_sim sub", result)
    assert (result == 2)

    result = yield from run_op(dut, 13, 2, MicrOp.OP_SHR)
    print("alu_sim shr", result)
    assert (result == 3)


def test_alu():
    alu = ALU(width=16)
    write_alu_gtkw("test_alusim.gtkw", clk_period=10e-9)
    run_simulation(alu, {"sync": alu_sim(alu)}, vcd_name='test_alusim.vcd')

    vl = rtlil.convert(alu, ports=alu.ports())
    with open("test_alu.il", "w") as f:
        f.write(vl)


def test_alu_parallel():
    # Compare with the sequential test implementation, above.
    m = Module()
    m.submodules.alu = dut = ALU(width=16)
    write_alu_gtkw("test_alu_parallel.gtkw", sub_module='alu',
                   pysim=is_engine_pysim())

    sim = Simulator(m)
    sim.add_clock(1e-6)

    def send(a, b, op, inv_a=0, rc=0):
        # present input data and assert i_valid
        yield dut.a.eq(a)
        yield dut.b.eq(b)
        yield dut.op.insn_type.eq(op)
        yield dut.op.invert_in.eq(inv_a)
        yield dut.op.rc.rc.eq(rc)
        yield dut.p.i_valid.eq(1)
        yield
        # wait for o_ready to be asserted
        while not (yield dut.p.o_ready):
            yield
        # clear input data and negate i_valid
        # if send is called again immediately afterwards, there will be no
        # visible transition (they will not be negated, after all)
        yield dut.p.i_valid.eq(0)
        yield dut.a.eq(0)
        yield dut.b.eq(0)
        yield dut.op.insn_type.eq(0)
        yield dut.op.invert_in.eq(0)
        yield dut.op.rc.rc.eq(0)

    def receive():
        # signal readiness to receive data
        yield dut.n.i_ready.eq(1)
        yield
        # wait for o_valid to be asserted
        while not (yield dut.n.o_valid):
            yield
        # read results
        result = yield dut.o.data
        cr = yield dut.cr.data
        # negate i_ready
        # if receive is called again immediately afterwards, there will be no
        # visible transition (it will not be negated, after all)
        yield dut.n.i_ready.eq(0)
        return result, cr

    def producer():
        # send a few test cases, interspersed with wait states
        # note that, for this test, we do not wait for the result to be ready,
        # before presenting the next input
        # 5 + 3
        yield from send(5, 3, MicrOp.OP_ADD)
        yield
        yield
        # 2 * 3
        yield from send(2, 3, MicrOp.OP_MUL_L64, rc=1)
        # (-6) + 3
        yield from send(5, 3, MicrOp.OP_ADD, inv_a=1, rc=1)
        yield
        # 5 - 3
        # note that this is a zero-delay operation
        yield from send(5, 3, MicrOp.OP_CMP)
        yield
        yield
        # NOP
        yield from send(5, 3, MicrOp.OP_NOP)
        # 13 >> 2
        yield from send(13, 2, MicrOp.OP_SHR)
        # sign extent 13
        yield from send(13, 2, MicrOp.OP_EXTS)
        # sign extend -128 (8 bits)
        yield from send(0x80, 2, MicrOp.OP_EXTS, rc=1)
        # sign extend -128 (8 bits)
        yield from send(2, 0x80, MicrOp.OP_EXTSWSLI)
        # 5 - 5
        yield from send(5, 5, MicrOp.OP_CMP, rc=1)

    def consumer():
        # receive and check results, interspersed with wait states
        # the consumer is not in step with the producer, but the
        # order of the results are preserved
        yield
        # 5 + 3 = 8
        result = yield from receive()
        assert result[0] == 8
        # 2 * 3 = 6
        # 6 > 0 => CR = 0b100
        result = yield from receive()
        assert result == (6, 0b100)
        yield
        yield
        # (-6) + 3 = -3
        # -3 < 0 => CR = 0b010
        result = yield from receive()
        assert result == (65533, 0b010)  # unsigned equivalent to -2
        # 5 - 3 = 2
        # note that this is a zero-delay operation
        # this, and the previous result, will be received back-to-back
        # (check the output waveform to see this)
        result = yield from receive()
        assert result[0] == 2
        yield
        yield
        # NOP
        yield from receive()
        # 13 >> 2 = 3
        result = yield from receive()
        assert result[0] == 3
        # sign extent 13 = 13
        result = yield from receive()
        assert result[0] == 13
        # sign extend -128 (8 bits) = -128 (16 bits)
        # -128 < 0 => CR = 0b010
        result = yield from receive()
        assert result == (0xFF80, 0b010)
        # sign extend -128 (8 bits) = -128 (16 bits)
        result = yield from receive()
        assert result[0] == 0xFF80
        # 5 - 5 = 0
        # 0 == 0 => CR = 0b001
        result = yield from receive()
        assert result == (0, 0b001)

    sim.add_sync_process(producer)
    sim.add_sync_process(consumer)
    sim_writer = sim.write_vcd("test_alu_parallel.vcd")
    with sim_writer:
        sim.run()


def write_alu_gtkw(gtkw_name, clk_period=1e-6, sub_module=None,
                   pysim=True):
    """Common function to write the GTKWave documents for this module"""
    gtkwave_desc = [
        'clk',
        'i1[15:0]',
        'i2[15:0]',
        'op__insn_type' if pysim else 'op__insn_type[6:0]',
        'op__invert_in',
        'i_valid',
        'o_ready',
        'o_valid',
        'i_ready',
        'alu_o[15:0]',
        'alu_o_ok',
        'alu_cr[15:0]',
        'alu_cr_ok'
    ]
    # determine the module name of the DUT
    module = 'top'
    if sub_module is not None:
        module = nmigen_sim_top_module + sub_module
    vcd_name = gtkw_name.replace('.gtkw', '.vcd')
    write_gtkw(gtkw_name, vcd_name, gtkwave_desc, module=module,
               loc=__file__, clk_period=clk_period, base='signed')


if __name__ == "__main__":
    test_alu()
    test_alu_parallel()

    # alu = BranchALU(width=16)
    # vl = rtlil.convert(alu, ports=alu.ports())
    # with open("test_branch_alu.il", "w") as f:
    #     f.write(vl)
