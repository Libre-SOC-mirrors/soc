# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2022 Cesar Strauss <cestrauss@gmail.com>
# Sponsored by NLnet and NGI POINTER under EU Grants 871528 and 957073
# Part of the Libre-SOC Project.

"""
Wrapper around a single port (1R or 1W) SRAM, to make a multi-port regfile.

This SRAM primitive has one cycle delay for reads, and, after a write,
it reads the value just written. The goal is to use it to make at least an
1W2R regfile.

See https://bugs.libre-soc.org/show_bug.cgi?id=781 and
https://bugs.libre-soc.org/show_bug.cgi?id=502
"""

import unittest

from nmigen import Elaboratable, Module, Memory, Signal
from nmigen.back import rtlil
from nmigen.sim import Simulator

from nmutil.formaltest import FHDLTestCase
from nmutil.gtkw import write_gtkw


class SinglePortSRAM(Elaboratable):
    """
    Model of a single port SRAM, which can be simulated, verified and/or
    synthesized to an FPGA.

    :param addr_width: width of the address bus
    :param data_width: width of the data bus
    :param we_width: number of write enable lines
    """
    def __init__(self, addr_width, data_width, we_width):
        self.addr_width = addr_width
        self.data_width = data_width
        self.we_width = we_width
        self.d = Signal(data_width)
        """ write data"""
        self.q = Signal(data_width)
        """read data"""
        self.a = Signal(addr_width)
        """ read/write address"""
        self.we = Signal(we_width)
        """write enable"""

    def elaborate(self, _):
        m = Module()
        # backing memory
        depth = 1 << self.addr_width
        granularity = self.data_width // self.we_width
        mem = Memory(width=self.data_width, depth=depth)
        # create read and write ports
        # By connecting the same address to both ports, they behave, in fact,
        # as a single, "half-duplex" port.
        # The transparent attribute means that, on a write, we read the new
        # value, on the next cycle
        # Note that nmigen memories have a one cycle delay, for reads,
        # by default
        m.submodules.rdport = rdport = mem.read_port(transparent=True)
        m.submodules.wrport = wrport = mem.write_port(granularity=granularity)
        # duplicate the address to both ports
        m.d.comb += wrport.addr.eq(self.a)
        m.d.comb += rdport.addr.eq(self.a)
        # write enable
        m.d.comb += wrport.en.eq(self.we)
        # read and write data
        m.d.comb += wrport.data.eq(self.d)
        m.d.comb += self.q.eq(rdport.data)
        return m

    def ports(self):
        return [
            self.d,
            self.a,
            self.we,
            self.q
        ]


def create_ilang(dut, ports, test_name):
    vl = rtlil.convert(dut, name=test_name, ports=ports)
    with open("%s.il" % test_name, "w") as f:
        f.write(vl)


class SinglePortSRAMTestCase(FHDLTestCase):
    @staticmethod
    def test_simple_rtlil():
        """
        Generate a simple SRAM. Try ``read_rtlil mem_simple.il; proc; show``
        from a yosys prompt, to see the memory primitives, and
        ``read_rtlil mem_simple.il; synth; show`` to see it implemented as
        flip-flop RAM
        """
        dut = SinglePortSRAM(2, 4, 2)
        create_ilang(dut, dut.ports(), "mem_simple")

    @staticmethod
    def test_blkram_rtlil():
        """
        Generates a bigger SRAM.
        Try ``read_rtlil mem_blkram.il; synth_ecp5; show`` from a yosys
        prompt, to see it implemented as block RAM
        """
        dut = SinglePortSRAM(10, 16, 2)
        create_ilang(dut, dut.ports(), "mem_blkram")

    def test_sram_model(self):
        """
        Simulate some read/write/modify operations on the SRAM model
        """
        dut = SinglePortSRAM(7, 32, 4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)

        def process():
            # 1) write 0x12_34_56_78 to address 0
            yield dut.a.eq(0)
            yield dut.d.eq(0x12_34_56_78)
            yield dut.we.eq(0b1111)
            yield
            # 2) write 0x9A_BC_DE_F0 to address 1
            yield dut.a.eq(1)
            yield dut.d.eq(0x9A_BC_DE_F0)
            yield dut.we.eq(0b1111)
            yield
            # ... and read value just written to address 0
            self.assertEqual((yield dut.q), 0x12_34_56_78)
            # 3) prepare to read from address 0
            yield dut.d.eq(0)
            yield dut.we.eq(0b0000)
            yield dut.a.eq(0)
            yield
            # ... and read value just written to address 1
            self.assertEqual((yield dut.q), 0x9A_BC_DE_F0)
            # 4) prepare to read from address 1
            yield dut.a.eq(1)
            yield
            # ... and read value from address 0
            self.assertEqual((yield dut.q), 0x12_34_56_78)
            # 5) write 0x9A and 0xDE to bytes 1 and 3, leaving
            # bytes 0 and 2 unchanged
            yield dut.a.eq(0)
            yield dut.d.eq(0x9A_FF_DE_FF)
            yield dut.we.eq(0b1010)
            yield
            # ... and read value from address 1
            self.assertEqual((yield dut.q), 0x9A_BC_DE_F0)
            # 6) nothing more to do
            yield dut.d.eq(0)
            yield dut.we.eq(0)
            yield
            # ... other than confirm that bytes 1 and 3 were modified
            # correctly
            self.assertEqual((yield dut.q), 0x9A_34_DE_78)

        sim.add_sync_process(process)
        traces = ['rdport.clk', 'a[6:0]', 'we[3:0]', 'd[31:0]', 'q[31:0]']
        write_gtkw('test_sram_model.gtkw', 'test_sram_model.vcd',
                   traces, module='top')
        sim_writer = sim.write_vcd('test_sram_model.vcd')
        with sim_writer:
            sim.run()


if __name__ == "__main__":
    unittest.main()
