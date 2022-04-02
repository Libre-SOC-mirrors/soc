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
from nmigen.asserts import Assert, Past, AnyConst

from nmutil.formaltest import FHDLTestCase
from nmutil.gtkw import write_gtkw


class SinglePortSRAM(Elaboratable):
    """
    Model of a single port SRAM, which can be simulated, verified and/or
    synthesized to an FPGA.

    :param addr_width: width of the address bus
    :param data_width: width of the data bus
    :param we_width: number of write enable lines

    .. note:: The debug read port is meant only to assist in formal proofs!
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
        self.dbg_a = Signal(addr_width)
        """debug read port address"""
        self.dbg_q = Signal(data_width)
        """debug read port data"""

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
        # the debug port is an asynchronous read port, allowing direct access
        # to a given memory location by the formal engine
        m.submodules.dbgport = dbgport = mem.read_port(domain="comb")
        m.d.comb += dbgport.addr.eq(self.dbg_a)
        m.d.comb += self.dbg_q.eq(dbgport.data)
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

    def test_model_sram_proof(self):
        """
        Formal proof of the single port SRAM model
        """
        m = Module()
        # 128 x 32-bit, 8-bit granularity
        m.submodules.dut = dut = SinglePortSRAM(7, 32, 4)
        gran = len(dut.d) // len(dut.we)  # granularity
        # choose a single random memory location to test
        a_const = AnyConst(dut.a.shape())
        # choose a single byte lane to test (one-hot encoding)
        we_mask = Signal.like(dut.we)
        # ... by first creating a random bit pattern
        we_const = AnyConst(dut.we.shape())
        # ... and zeroing all but the first non-zero bit
        m.d.comb += we_mask.eq(we_const & (-we_const))
        # holding data register
        d_reg = Signal(gran)
        # for some reason, simulated formal memory is not zeroed at reset
        # ... so, remember whether we wrote it, at least once.
        wrote = Signal()
        # if our memory location and byte lane is being written
        # ... capture the data in our holding register
        with m.If(dut.a == a_const):
            for i in range(len(dut.we)):
                with m.If(we_mask[i] & dut.we[i]):
                    m.d.sync += d_reg.eq(dut.d[i*gran:i*gran+gran])
                    m.d.sync += wrote.eq(1)
        # if our memory location is being read
        # ... and the holding register has valid data
        # ... then its value must match the memory output, on the given lane
        with m.If((Past(dut.a) == a_const) & wrote):
            for i in range(len(dut.we)):
                with m.If(we_mask[i]):
                    m.d.sync += Assert(d_reg == dut.q[i*gran:i*gran+gran])

        # the following is needed for induction, where an unreachable state
        # (memory and holding register differ) is turned into an illegal one
        # first, get the value stored in our memory location, using its debug
        # port
        stored = Signal.like(dut.q)
        m.d.comb += dut.dbg_a.eq(a_const)
        m.d.comb += stored.eq(dut.dbg_q)
        # now, ensure that the value stored in memory is always in sync
        # with the holding register
        with m.If(wrote):
            for i in range(len(dut.we)):
                with m.If(we_mask[i]):
                    m.d.sync += Assert(d_reg == stored[i*gran:i*gran+gran])

        self.assertFormal(m, mode="prove", depth=2)


class PhasedDualPortRegfile(Elaboratable):
    """
    Builds, from a pair of 1RW blocks, a pseudo 1W/1R RAM, where the
    read port works every cycle, but the write port is only available on
    either even (1eW/1R) or odd (1oW/1R) cycles.

    :param addr_width: width of the address bus
    :param data_width: width of the data bus
    :param we_width: number of write enable lines
    :param write_phase: indicates on which phase the write port will
                        accept data
    :param transparent: whether a simultaneous read and write returns the
                        new value (True) or the old value (False)
    """

    def __init__(self, addr_width, data_width, we_width, write_phase,
                 transparent=False):
        self.addr_width = addr_width
        self.data_width = data_width
        self.we_width = we_width
        self.write_phase = write_phase
        self.transparent = transparent
        self.wr_addr_i = Signal(addr_width)
        """write port address"""
        self.wr_data_i = Signal(data_width)
        """write port data"""
        self.wr_we_i = Signal(we_width)
        """write port enable"""
        self.rd_addr_i = Signal(addr_width)
        """read port address"""
        self.rd_data_o = Signal(data_width)
        """read port data"""
        self.phase = Signal()
        """even/odd cycle indicator"""

    def elaborate(self, _):
        m = Module()
        # instantiate the two 1RW memory blocks
        mem1 = SinglePortSRAM(self.addr_width, self.data_width, self.we_width)
        mem2 = SinglePortSRAM(self.addr_width, self.data_width, self.we_width)
        m.submodules.mem1 = mem1
        m.submodules.mem2 = mem2
        # wire write port to first memory, and its output to the second
        m.d.comb += mem1.d.eq(self.wr_data_i)
        m.d.comb += mem2.d.eq(mem1.q)
        # holding registers for the write port of the second memory
        last_wr_addr = Signal(self.addr_width)
        last_wr_we = Signal(self.we_width)
        # do the read and write address coincide?
        same_read_write = Signal()
        with m.If(self.phase == self.write_phase):
            # write phase, start a write on the first memory
            m.d.comb += mem1.a.eq(self.wr_addr_i)
            m.d.comb += mem1.we.eq(self.wr_we_i)
            # save write address and write select for repeating the write
            # on the second memory, later
            m.d.sync += last_wr_we.eq(self.wr_we_i)
            m.d.sync += last_wr_addr.eq(self.wr_addr_i)
            # start a read on the second memory
            m.d.comb += mem2.a.eq(self.rd_addr_i)
            # output previously read data from the first memory
            m.d.comb += self.rd_data_o.eq(mem1.q)
            if self.transparent:
                # remember whether we are reading from the same location we are
                # writing
                m.d.sync += same_read_write.eq(self.rd_addr_i == self.wr_addr_i)
        with m.Else():
            # read phase, write last written data on second memory
            m.d.comb += mem2.a.eq(last_wr_addr)
            m.d.comb += mem2.we.eq(last_wr_we)
            # start a read on the first memory
            m.d.comb += mem1.a.eq(self.rd_addr_i)
            if self.transparent:
                with m.If(same_read_write):
                    # when transparent, and read and write addresses coincide,
                    # output the data just written
                    m.d.comb += self.rd_data_o.eq(mem1.q)
                with m.Else():
                    # otherwise, output previously read data
                    # from the second memory
                    m.d.comb += self.rd_data_o.eq(mem2.q)
            else:
                # always output the read data from the second memory,
                # if not transparent
                m.d.comb += self.rd_data_o.eq(mem2.q)

        return m


class PhasedDualPortRegfileTestCase(FHDLTestCase):

    def do_test_phased_dual_port_regfile(self, write_phase, transparent):
        """
        Simulate some read/write/modify operations on the phased write memory
        """
        dut = PhasedDualPortRegfile(7, 32, 4, write_phase, transparent)
        sim = Simulator(dut)
        sim.add_clock(1e-6)

        # compare read data with previously written data
        # and start a new read
        def read(rd_addr_i, expected=None):
            if expected is not None:
                self.assertEqual((yield dut.rd_data_o), expected)
            yield dut.rd_addr_i.eq(rd_addr_i)

        # start a write, and set write phase
        def write(wr_addr_i, wr_we_i, wr_data_i):
            yield dut.wr_addr_i.eq(wr_addr_i)
            yield dut.wr_we_i.eq(wr_we_i)
            yield dut.wr_data_i.eq(wr_data_i)
            yield dut.phase.eq(write_phase)

        # disable writes, and start read phase
        def skip_write():
            yield dut.wr_addr_i.eq(0)
            yield dut.wr_we_i.eq(0)
            yield dut.wr_data_i.eq(0)
            yield dut.phase.eq(~write_phase)

        # writes a few values on the write port, and read them back
        # ... reads can happen every cycle
        # ... writes, only every two cycles.
        # since reads have a one cycle delay, the expected value on
        # each read refers to the last read performed, not the
        # current one, which is in progress.
        def process():
            yield from read(0)
            yield from write(0x42, 0b1111, 0x12345678)
            yield
            yield from read(0x42)
            yield from skip_write()
            yield
            yield from read(0x42)
            yield from write(0x43, 0b1111, 0x9ABCDEF0)
            yield
            yield from read(0x43, 0x12345678)
            yield from skip_write()
            yield
            yield from read(0x42, 0x12345678)
            yield from write(0x43, 0b1001, 0xF0FFFF9A)
            yield
            yield from read(0x43, 0x9ABCDEF0)
            yield from skip_write()
            yield
            yield from read(0x43, 0x12345678)
            yield from write(0x42, 0b0110, 0xFF5634FF)
            yield
            yield from read(0x42, 0xF0BCDE9A)
            yield from skip_write()
            yield
            yield from read(0, 0xF0BCDE9A)
            yield from write(0, 0, 0)
            yield
            yield from read(0, 0x12563478)
            yield from skip_write()
            yield
            # try reading and writing to the same location, simultaneously
            yield from read(0x42)
            yield from write(0x42, 0b1111, 0x55AA9966)
            yield
            # ... and read again
            yield from read(0x42)
            yield from skip_write()
            yield
            if transparent:
                # returns the value just written
                yield from read(0, 0x55AA9966)
            else:
                # returns the old value
                yield from read(0, 0x12563478)
            yield from write(0, 0, 0)
            yield
            # after a cycle, always returns the new value
            yield from read(0, 0x55AA9966)
            yield from skip_write()

        sim.add_sync_process(process)
        debug_file = f'test_phased_dual_port_{write_phase}'
        if transparent:
            debug_file += '_transparent'
        traces = ['clk', 'phase',
                  'wr_addr_i[6:0]', 'wr_we_i[3:0]', 'wr_data_i[31:0]',
                  'rd_addr_i[6:0]', 'rd_data_o[31:0]']
        write_gtkw(debug_file + '.gtkw',
                   debug_file + '.vcd',
                   traces, module='top', zoom=-22)
        sim_writer = sim.write_vcd(debug_file + '.vcd')
        with sim_writer:
            sim.run()

    def test_phased_dual_port_regfile(self):
        """test both types (odd and even write ports) of phased write memory"""
        with self.subTest("writes happen on phase 0"):
            self.do_test_phased_dual_port_regfile(0, False)
        with self.subTest("writes happen on phase 1"):
            self.do_test_phased_dual_port_regfile(1, False)
        """test again, with a transparent read port"""
        with self.subTest("writes happen on phase 0 (transparent reads)"):
            self.do_test_phased_dual_port_regfile(0, True)
        with self.subTest("writes happen on phase 1 (transparent reads)"):
            self.do_test_phased_dual_port_regfile(1, True)


if __name__ == "__main__":
    unittest.main()
