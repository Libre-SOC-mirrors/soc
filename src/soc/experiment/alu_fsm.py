"""Simple example of a FSM-based ALU

This demonstrates a design that follows the valid/ready protocol of the
ALU, but with a FSM implementation, instead of a pipeline.  It is also
intended to comply with both the CompALU API and the nmutil Pipeline API
(Liskov Substitution Principle)

The basic rules are:

1) p.ready_o is asserted on the initial ("Idle") state, otherwise it keeps low.
2) n.valid_o is asserted on the final ("Done") state, otherwise it keeps low.
3) The FSM stays in the Idle state while p.valid_i is low, otherwise
   it accepts the input data and moves on.
4) The FSM stays in the Done state while n.ready_i is low, otherwise
   it releases the output data and goes back to the Idle state.

"""

from nmigen import Elaboratable, Signal, Module, Cat
cxxsim = False
if cxxsim:
    from nmigen.sim.cxxsim import Simulator, Settle
else:
    from nmigen.back.pysim import Simulator, Settle
from nmigen.cli import rtlil
from math import log2
from nmutil.iocontrol import PrevControl, NextControl

from soc.fu.base_input_record import CompOpSubsetBase
from soc.decoder.power_enums import (MicrOp, Function)

from nmutil.gtkw import write_gtkw


class CompFSMOpSubset(CompOpSubsetBase):
    def __init__(self, name=None):
        layout = (('sdir', 1),
                  )

        super().__init__(layout, name=name)



class Dummy:
    pass


class Shifter(Elaboratable):
    """Simple sequential shifter

    Prev port data:
    * p.data_i.data:  value to be shifted
    * p.data_i.shift: shift amount
    *                 When zero, no shift occurs.
    *                 On POWER, range is 0 to 63 for 32-bit,
    *                 and 0 to 127 for 64-bit.
    *                 Other values wrap around.

    Operation type
    * op.sdir:       shift direction (0 = left, 1 = right)

    Next port data:
    * n.data_o.data: shifted value
    """
    class PrevData:
        def __init__(self, width):
            self.data = Signal(width, name="p_data_i")
            self.shift = Signal(width, name="p_shift_i")
            self.ctx = Dummy() # comply with CompALU API

        def _get_data(self):
            return [self.data, self.shift]

    class NextData:
        def __init__(self, width):
            self.data = Signal(width, name="n_data_o")

        def _get_data(self):
            return [self.data]

    def __init__(self, width):
        self.width = width
        self.p = PrevControl()
        self.n = NextControl()
        self.p.data_i = Shifter.PrevData(width)
        self.n.data_o = Shifter.NextData(width)

        # more pieces to make this example class comply with the CompALU API
        self.op = CompFSMOpSubset(name="op")
        self.p.data_i.ctx.op = self.op
        self.i = self.p.data_i._get_data()
        self.out = self.n.data_o._get_data()

    def elaborate(self, platform):
        m = Module()

        m.submodules.p = self.p
        m.submodules.n = self.n

        # Note:
        # It is good practice to design a sequential circuit as
        # a data path and a control path.

        # Data path
        # ---------
        # The idea is to have a register that can be
        # loaded or shifted (left and right).

        # the control signals
        load = Signal()
        shift = Signal()
        direction = Signal()
        # the data flow
        shift_in = Signal(self.width)
        shift_left_by_1 = Signal(self.width)
        shift_right_by_1 = Signal(self.width)
        next_shift = Signal(self.width)
        # the register
        shift_reg = Signal(self.width, reset_less=True)
        # build the data flow
        m.d.comb += [
            # connect input and output
            shift_in.eq(self.p.data_i.data),
            self.n.data_o.data.eq(shift_reg),
            # generate shifted views of the register
            shift_left_by_1.eq(Cat(0, shift_reg[:-1])),
            shift_right_by_1.eq(Cat(shift_reg[1:], 0)),
        ]
        # choose the next value of the register according to the
        # control signals
        # default is no change
        m.d.comb += next_shift.eq(shift_reg)
        with m.If(load):
            m.d.comb += next_shift.eq(shift_in)
        with m.Elif(shift):
            with m.If(direction):
                m.d.comb += next_shift.eq(shift_right_by_1)
            with m.Else():
                m.d.comb += next_shift.eq(shift_left_by_1)

        # register the next value
        m.d.sync += shift_reg.eq(next_shift)

        # Control path
        # ------------
        # The idea is to have a SHIFT state where the shift register
        # is shifted every cycle, while a counter decrements.
        # This counter is loaded with shift amount in the initial state.
        # The SHIFT state is left when the counter goes to zero.

        # Shift counter
        shift_width = int(log2(self.width)) + 1
        next_count = Signal(shift_width)
        count = Signal(shift_width, reset_less=True)
        m.d.sync += count.eq(next_count)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += [
                    # keep p.ready_o active on IDLE
                    self.p.ready_o.eq(1),
                    # keep loading the shift register and shift count
                    load.eq(1),
                    next_count.eq(self.p.data_i.shift),
                ]
                # capture the direction bit as well
                m.d.sync += direction.eq(self.op.sdir)
                with m.If(self.p.valid_i):
                    # Leave IDLE when data arrives
                    with m.If(next_count == 0):
                        # short-circuit for zero shift
                        m.next = "DONE"
                    with m.Else():
                        m.next = "SHIFT"
            with m.State("SHIFT"):
                m.d.comb += [
                    # keep shifting, while counter is not zero
                    shift.eq(1),
                    # decrement the shift counter
                    next_count.eq(count - 1),
                ]
                with m.If(next_count == 0):
                    # exit when shift counter goes to zero
                    m.next = "DONE"
            with m.State("DONE"):
                # keep n.valid_o active while the data is not accepted
                m.d.comb += self.n.valid_o.eq(1)
                with m.If(self.n.ready_i):
                    # go back to IDLE when the data is accepted
                    m.next = "IDLE"

        return m

    def __iter__(self):
        yield self.op.sdir
        yield self.p.data_i.data
        yield self.p.data_i.shift
        yield self.p.valid_i
        yield self.p.ready_o
        yield self.n.ready_i
        yield self.n.valid_o
        yield self.n.data_o.data

    def ports(self):
        return list(self)


def test_shifter():
    m = Module()
    m.submodules.shf = dut = Shifter(8)
    print("Shifter port names:")
    for port in dut:
        print("-", port.name)
    # generate RTLIL
    # try "proc; show" in yosys to check the data path
    il = rtlil.convert(dut, ports=dut.ports())
    with open("test_shifter.il", "w") as f:
        f.write(il)

    # Describe a GTKWave document

    # Style for signals, classes and groups
    gtkwave_style = {
        # Root selector. Gives default attributes for every signal.
        '': {'base': 'dec'},
        # color the traces, according to class
        # class names are not hardcoded, they are just strings
        'in': {'color': 'orange'},
        'out': {'color': 'yellow'},
        # signals in the debug group have a common color and module path
        'debug': {'module': 'top', 'color': 'red'},
        # display a different string replacing the signal name
        'test_case': {'display': 'test case'},
    }

    # DOM style description for the trace pane
    gtkwave_desc = [
        # simple signal, without a class
        # even so, it inherits the top-level root attributes
        'clk',
        # comment
        {'comment': 'Shifter Demonstration'},
        # collapsible signal group
        ('prev port', [
            # attach a class style for each signal
            ('op__sdir', 'in'),
            ('p_data_i[7:0]', 'in'),
            ('p_shift_i[7:0]', 'in'),
            ('p_valid_i', 'in'),
            ('p_ready_o', 'out'),
        ]),
        # Signals in a signal group inherit the group attributes.
        # In this case, a different module path and color.
        ('debug', [
            {'comment': 'Some debug statements'},
            # inline attributes, instead of a class name
            ('zero', {'display': 'zero delay shift'}),
            'interesting',
            'test_case',
            'msg',
        ]),
        ('internal', [
            'fsm_state',
            'count[3:0]',
            'shift_reg[7:0]',
        ]),
        ('next port', [
            ('n_data_o[7:0]', 'out'),
            ('n_valid_o', 'out'),
            ('n_ready_i', 'in'),
        ]),
    ]

    write_gtkw("test_shifter.gtkw", "test_shifter.vcd",
               gtkwave_desc,  gtkwave_style,
               module="top.shf", loc=__file__, marker=10500000)

    sim = Simulator(m)
    sim.add_clock(1e-6)

    # demonstrates adding extra debug signal traces
    # they end up in the top module
    #
    zero = Signal()  # mark an interesting place
    #
    # demonstrates string traces
    #
    # display a message when the signal is high
    # the low level is just an horizontal line
    interesting = Signal(decoder=lambda v: 'interesting!' if v else '')
    # choose between alternate strings based on numerical value
    test_cases = ['', '13>>2', '3<<4', '21<<0']
    test_case = Signal(8, decoder=lambda v: test_cases[v])
    # hack to display arbitrary strings, like debug statements
    msg = Signal(decoder=lambda _: msg.str)
    msg.str = ''

    def send(data, shift, direction):
        # present input data and assert valid_i
        yield dut.p.data_i.data.eq(data)
        yield dut.p.data_i.shift.eq(shift)
        yield dut.op.sdir.eq(direction)
        yield dut.p.valid_i.eq(1)
        yield
        # wait for p.ready_o to be asserted
        while not (yield dut.p.ready_o):
            yield
        # show current operation operation
        if direction:
            msg.str = f'{data}>>{shift}'
        else:
            msg.str = f'{data}<<{shift}'
        # force dump of the above message by toggling the
        # underlying signal
        yield msg.eq(0)
        yield msg.eq(1)
        # clear input data and negate p.valid_i
        yield dut.p.valid_i.eq(0)
        yield dut.p.data_i.data.eq(0)
        yield dut.p.data_i.shift.eq(0)
        yield dut.op.sdir.eq(0)

    def receive(expected):
        # signal readiness to receive data
        yield dut.n.ready_i.eq(1)
        yield
        # wait for n.valid_o to be asserted
        while not (yield dut.n.valid_o):
            yield
        # read result
        result = yield dut.n.data_o.data
        # negate n.ready_i
        yield dut.n.ready_i.eq(0)
        # check result
        assert result == expected
        # finish displaying the current operation
        msg.str = ''
        yield msg.eq(0)
        yield msg.eq(1)

    def producer():
        # 13 >> 2
        yield from send(13, 2, 1)
        # 3 << 4
        yield from send(3, 4, 0)
        # 21 << 0
        # use a debug signal to mark an interesting operation
        # in this case, it is a shift by zero
        yield interesting.eq(1)
        yield from send(21, 0, 0)
        yield interesting.eq(0)

    def consumer():
        # the consumer is not in step with the producer, but the
        # order of the results are preserved
        # 13 >> 2 = 3
        yield test_case.eq(1)
        yield from receive(3)
        # 3 << 4 = 48
        yield test_case.eq(2)
        yield from receive(48)
        # 21 << 0 = 21
        yield test_case.eq(3)
        # you can look for the rising edge of this signal to quickly
        # locate this point in the traces
        yield zero.eq(1)
        yield from receive(21)
        yield zero.eq(0)
        yield test_case.eq(0)

    sim.add_sync_process(producer)
    sim.add_sync_process(consumer)
    sim_writer = sim.write_vcd(
        "test_shifter.vcd",
        # include additional signals in the trace dump
        traces=[zero, interesting, test_case, msg],
    )
    with sim_writer:
        sim.run()


if __name__ == "__main__":
    test_shifter()
