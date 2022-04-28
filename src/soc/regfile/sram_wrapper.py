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

from nmigen import Elaboratable, Module, Memory, Signal, Repl, Mux
from nmigen.back import rtlil
from nmigen.sim import Simulator
from nmigen.asserts import Assert, Assume, Past, AnyConst

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
        # interface signals
        self.d = Signal(data_width); """ write data"""
        self.q = Signal(data_width); """read data"""
        self.a = Signal(addr_width); """ read/write address"""
        self.we = Signal(we_width); """write enable"""
        # debug signals, only used in formal proofs
        self.dbg_addr = Signal(addr_width); """debug: address under test"""
        lanes = range(we_width)
        self.dbg_lane = Signal(lanes); """debug: write lane under test"""
        gran = self.data_width // self.we_width
        self.dbg_data = Signal(gran); """debug: data to keep in sync"""
        self.dbg_wrote = Signal(); """debug: data is valid"""

    def elaborate(self, platform):
        m = Module()
        # backing memory
        depth = 1 << self.addr_width
        gran = self.data_width // self.we_width
        mem = Memory(width=self.data_width, depth=depth)
        # create read and write ports
        # By connecting the same address to both ports, they behave, in fact,
        # as a single, "half-duplex" port.
        # The transparent attribute means that, on a write, we read the new
        # value, on the next cycle
        # Note that nmigen memories have a one cycle delay, for reads,
        # by default
        m.submodules.rdport = rdport = mem.read_port(transparent=True)
        m.submodules.wrport = wrport = mem.write_port(granularity=gran)
        # duplicate the address to both ports
        m.d.comb += wrport.addr.eq(self.a)
        m.d.comb += rdport.addr.eq(self.a)
        # write enable
        m.d.comb += wrport.en.eq(self.we)
        # read and write data
        m.d.comb += wrport.data.eq(self.d)
        m.d.comb += self.q.eq(rdport.data)

        # the following is needed for induction, where an unreachable state
        # (memory and holding register differ) is turned into an illegal one
        if platform == "formal":
            # the debug port is an asynchronous read port, allowing direct
            # access to a given memory location by the formal engine
            m.submodules.dbgport = dbgport = mem.read_port(domain="comb")
            # first, get the value stored in our memory location,
            # using its debug port
            stored = Signal(self.data_width)
            m.d.comb += dbgport.addr.eq(self.dbg_addr)
            m.d.comb += stored.eq(dbgport.data)
            # now, ensure that the value stored in memory is always in sync
            # with the holding register
            with m.If(self.dbg_wrote):
                m.d.sync += Assert(self.dbg_data ==
                                   stored.word_select(self.dbg_lane, gran))

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
        # choose a single byte lane to test
        lane = AnyConst(range(dut.we_width))
        # holding data register
        d_reg = Signal(gran)
        # for some reason, simulated formal memory is not zeroed at reset
        # ... so, remember whether we wrote it, at least once.
        wrote = Signal()
        # if our memory location and byte lane is being written
        # ... capture the data in our holding register
        with m.If((dut.a == a_const) & dut.we.bit_select(lane, 1)):
            m.d.sync += d_reg.eq(dut.d.word_select(lane, gran))
            m.d.sync += wrote.eq(1)
        # if our memory location is being read
        # ... and the holding register has valid data
        # ... then its value must match the memory output, on the given lane
        with m.If((Past(dut.a) == a_const) & wrote):
            m.d.sync += Assert(d_reg == dut.q.word_select(lane, gran))

        # pass our state to the device under test, so it can ensure that
        # its state is in sync with ours, for induction
        m.d.comb += [
            dut.dbg_addr.eq(a_const),
            dut.dbg_lane.eq(lane),
            dut.dbg_data.eq(d_reg),
            dut.dbg_wrote.eq(wrote),
        ]

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

    .. note:: The debug read port is meant only to assist in formal proofs!
    """

    def __init__(self, addr_width, data_width, we_width, write_phase,
                 transparent=False):
        self.addr_width = addr_width
        self.data_width = data_width
        self.we_width = we_width
        self.write_phase = write_phase
        self.transparent = transparent
        # interface signals
        self.wr_addr_i = Signal(addr_width); """write port address"""
        self.wr_data_i = Signal(data_width); """write port data"""
        self.wr_we_i = Signal(we_width); """write port enable"""
        self.rd_addr_i = Signal(addr_width); """read port address"""
        self.rd_data_o = Signal(data_width); """read port data"""
        self.phase = Signal(); """even/odd cycle indicator"""
        # debug signals, only used in formal proofs
        self.dbg_addr = Signal(addr_width); """debug: address under test"""
        lanes = range(we_width)
        self.dbg_lane = Signal(lanes); """debug: write lane under test"""
        gran = self.data_width // self.we_width
        self.dbg_data = Signal(gran); """debug: data to keep in sync"""
        self.dbg_wrote = Signal(); """debug: data is valid"""

    def elaborate(self, platform):
        m = Module()
        # granularity
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

        if platform == "formal":
            # pass our state to the device under test, so it can ensure that
            # its state is in sync with ours, for induction
            m.d.comb += [
                # pass the address and write lane under test to both memories
                mem1.dbg_addr.eq(self.dbg_addr),
                mem2.dbg_addr.eq(self.dbg_addr),
                mem1.dbg_lane.eq(self.dbg_lane),
                mem2.dbg_lane.eq(self.dbg_lane),
                # the second memory copies its state from the first memory,
                # after a cycle, so it has a one cycle delay
                mem1.dbg_data.eq(self.dbg_data),
                mem2.dbg_data.eq(Past(self.dbg_data)),
                mem1.dbg_wrote.eq(self.dbg_wrote),
                mem2.dbg_wrote.eq(Past(self.dbg_wrote)),
            ]

        return m

    def ports(self):
        return [
            self.wr_addr_i,
            self.wr_data_i,
            self.wr_we_i,
            self.rd_addr_i,
            self.rd_data_o,
            self.phase
        ]


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
            yield from write(0x42, 0b0101, 0x55AA9966)
            yield
            # ... and read again
            yield from read(0x42)
            yield from skip_write()
            yield
            if transparent:
                # returns the value just written
                yield from read(0, 0x12AA3466)
            else:
                # returns the old value
                yield from read(0, 0x12563478)
            yield from write(0, 0, 0)
            yield
            # after a cycle, always returns the new value
            yield from read(0, 0x12AA3466)
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

    def do_test_phased_dual_port_regfile_proof(self, write_phase, transparent):
        """
        Formal proof of the pseudo 1W/1R regfile
        """
        m = Module()
        # 128 x 32-bit, 8-bit granularity
        dut = PhasedDualPortRegfile(7, 32, 4, write_phase, transparent)
        m.submodules.dut = dut
        gran = dut.data_width // dut.we_width  # granularity
        # choose a single random memory location to test
        a_const = AnyConst(dut.addr_width)
        # choose a single byte lane to test
        lane = AnyConst(range(dut.we_width))
        # drive alternating phases
        m.d.comb += Assume(dut.phase != Past(dut.phase))
        # holding data register
        d_reg = Signal(gran)
        # for some reason, simulated formal memory is not zeroed at reset
        # ... so, remember whether we wrote it, at least once.
        wrote = Signal()
        # if our memory location and byte lane is being written,
        # capture the data in our holding register
        with m.If((dut.wr_addr_i == a_const)
                  & dut.wr_we_i.bit_select(lane, 1)
                  & (dut.phase == dut.write_phase)):
            m.d.sync += d_reg.eq(dut.wr_data_i.word_select(lane, gran))
            m.d.sync += wrote.eq(1)
        # if our memory location is being read,
        # and the holding register has valid data,
        # then its value must match the memory output, on the given lane
        with m.If(Past(dut.rd_addr_i) == a_const):
            if transparent:
                with m.If(wrote):
                    rd_lane = dut.rd_data_o.word_select(lane, gran)
                    m.d.sync += Assert(d_reg == rd_lane)
            else:
                # with a non-transparent read port, the read value depends
                # on whether there is a simultaneous write, or not
                with m.If((Past(dut.wr_addr_i) == a_const)
                          & Past(dut.phase) == dut.write_phase):
                    # simultaneous write -> check against last written value
                    with m.If(Past(wrote)):
                        rd_lane = dut.rd_data_o.word_select(lane, gran)
                        m.d.sync += Assert(Past(d_reg) == rd_lane)
                with m.Else():
                    # otherwise, check against current written value
                    with m.If(wrote):
                        rd_lane = dut.rd_data_o.word_select(lane, gran)
                        m.d.sync += Assert(d_reg == rd_lane)

        # pass our state to the device under test, so it can ensure that
        # its state is in sync with ours, for induction
        m.d.comb += [
            # address and mask under test
            dut.dbg_addr.eq(a_const),
            dut.dbg_lane.eq(lane),
            # state of our holding register
            dut.dbg_data.eq(d_reg),
            dut.dbg_wrote.eq(wrote),
        ]

        self.assertFormal(m, mode="prove", depth=3)

    def test_phased_dual_port_regfile_proof(self):
        """test both types (odd and even write ports) of phased write memory"""
        with self.subTest("writes happen on phase 0"):
            self.do_test_phased_dual_port_regfile_proof(0, False)
        with self.subTest("writes happen on phase 1"):
            self.do_test_phased_dual_port_regfile_proof(1, False)
        # test again, with transparent read ports
        with self.subTest("writes happen on phase 0 (transparent reads)"):
            self.do_test_phased_dual_port_regfile_proof(0, True)
        with self.subTest("writes happen on phase 1 (transparent reads)"):
            self.do_test_phased_dual_port_regfile_proof(1, True)


class DualPortRegfile(Elaboratable):
    """
    Builds, from a pair of phased 1W/1R blocks, a true 1W/1R RAM, where both
    read and write ports work every cycle.
    It employs a Last Value Table, that tracks to which memory each address was
    last written.

    :param addr_width: width of the address bus
    :param data_width: width of the data bus
    :param we_width: number of write enable lines
    :param transparent: whether a simultaneous read and write returns the
                        new value (True) or the old value (False)
    """

    def __init__(self, addr_width, data_width, we_width, transparent=True):
        self.addr_width = addr_width
        self.data_width = data_width
        self.we_width = we_width
        self.transparent = transparent
        # interface signals
        self.wr_addr_i = Signal(addr_width); """write port address"""
        self.wr_data_i = Signal(data_width); """write port data"""
        self.wr_we_i = Signal(we_width); """write port enable"""
        self.rd_addr_i = Signal(addr_width); """read port address"""
        self.rd_data_o = Signal(data_width); """read port data"""
        # debug signals, only used in formal proofs
        # address and write lane under test
        self.dbg_addr = Signal(addr_width); """debug: address under test"""
        lanes = range(we_width)
        self.dbg_lane = Signal(lanes); """debug: write lane under test"""
        # upstream state, to keep in sync with ours
        gran = self.data_width // self.we_width
        self.dbg_data = Signal(gran); """debug: data to keep in sync"""
        self.dbg_wrote = Signal(); """debug: data is valid"""
        self.dbg_wrote_phase = Signal(); """debug: the phase data was written"""
        self.dbg_phase = Signal(); """debug: current phase"""

    def elaborate(self, platform):
        m = Module()
        # depth and granularity
        depth = 1 << self.addr_width
        gran = self.data_width // self.we_width
        # instantiate the two phased 1R/1W memory blocks
        mem0 = PhasedDualPortRegfile(
            self.addr_width, self.data_width, self.we_width, 0,
            self.transparent)
        mem1 = PhasedDualPortRegfile(
            self.addr_width, self.data_width, self.we_width, 1,
            self.transparent)
        m.submodules.mem0 = mem0
        m.submodules.mem1 = mem1
        # instantiate the backing memory (FFRAM or LUTRAM)
        # for the Last Value Table
        # it should have the same number and port types of the desired
        # memory, but just one bit per write lane
        lvt_mem = Memory(width=self.we_width, depth=depth)
        lvt_wr = lvt_mem.write_port(granularity=1)
        lvt_rd = lvt_mem.read_port(transparent=self.transparent)
        if not self.transparent:
            # for some reason, formal proofs don't recognize the default
            # reset value for this signal
            m.d.comb += lvt_rd.en.eq(1)
        m.submodules.lvt_wr = lvt_wr
        m.submodules.lvt_rd = lvt_rd
        # generate and wire the phases for the phased memories
        phase = Signal()
        m.d.sync += phase.eq(~phase)
        m.d.comb += [
            mem0.phase.eq(phase),
            mem1.phase.eq(phase),
        ]
        m.d.comb += [
            # wire the write ports, directly
            mem0.wr_addr_i.eq(self.wr_addr_i),
            mem1.wr_addr_i.eq(self.wr_addr_i),
            mem0.wr_we_i.eq(self.wr_we_i),
            mem1.wr_we_i.eq(self.wr_we_i),
            mem0.wr_data_i.eq(self.wr_data_i),
            mem1.wr_data_i.eq(self.wr_data_i),
            # also wire the read addresses
            mem0.rd_addr_i.eq(self.rd_addr_i),
            mem1.rd_addr_i.eq(self.rd_addr_i),
            # wire read and write ports to the LVT
            lvt_wr.addr.eq(self.wr_addr_i),
            lvt_wr.en.eq(self.wr_we_i),
            lvt_rd.addr.eq(self.rd_addr_i),
            # the data for the LVT is the phase on which the value was
            # written
            lvt_wr.data.eq(Repl(phase, self.we_width)),
        ]
        for i in range(self.we_width):
            # select the right memory to assign to the output read port,
            # in this byte lane, according to the LVT contents
            m.d.comb += self.rd_data_o.word_select(i, gran).eq(
                Mux(
                    lvt_rd.data[i],
                    mem1.rd_data_o.word_select(i, gran),
                    mem0.rd_data_o.word_select(i, gran)))

        if platform == "formal":
            # pass upstream state to the memories, so they can ensure that
            # their state are in sync with upstream, for induction
            m.d.comb += [
                # address and write lane under test
                mem0.dbg_addr.eq(self.dbg_addr),
                mem1.dbg_addr.eq(self.dbg_addr),
                mem0.dbg_lane.eq(self.dbg_lane),
                mem1.dbg_lane.eq(self.dbg_lane),
                # upstream state
                mem0.dbg_data.eq(self.dbg_data),
                mem1.dbg_data.eq(self.dbg_data),
                # the memory, on which the write ends up, depends on which
                # phase it was written
                mem0.dbg_wrote.eq(self.dbg_wrote & ~self.dbg_wrote_phase),
                mem1.dbg_wrote.eq(self.dbg_wrote & self.dbg_wrote_phase),
            ]
            # sync phase to upstream
            m.d.comb += Assert(self.dbg_phase == phase)
            # this debug port for the LVT is an asynchronous read port,
            # allowing direct access to a given memory location
            # by the formal engine
            m.submodules.dbgport = dbgport = lvt_mem.read_port(domain='comb')
            # first, get the value stored in our memory location,
            stored = Signal(self.we_width)
            m.d.comb += dbgport.addr.eq(self.dbg_addr)
            m.d.comb += stored.eq(dbgport.data)
            # now, ensure that the value stored in memory is always in sync
            # with the expected value (which memory the value was written to)
            with m.If(self.dbg_wrote):
                m.d.comb += Assert(stored.bit_select(self.dbg_lane, 1)
                                   == self.dbg_wrote_phase)
        return m

    def ports(self):
        return [
            self.wr_addr_i,
            self.wr_data_i,
            self.wr_we_i,
            self.rd_addr_i,
            self.rd_data_o
        ]


class DualPortRegfileTestCase(FHDLTestCase):

    def do_test_dual_port_regfile(self, transparent):
        """
        Simulate some read/write/modify operations on the dual port register
        file
        """
        dut = DualPortRegfile(7, 32, 4, transparent)
        sim = Simulator(dut)
        sim.add_clock(1e-6)

        expected = None
        last_expected = None

        # compare read data with previously written data
        # and start a new read
        def read(rd_addr_i, next_expected=None):
            nonlocal expected, last_expected
            if expected is not None:
                self.assertEqual((yield dut.rd_data_o), expected)
            yield dut.rd_addr_i.eq(rd_addr_i)
            # account for the read latency
            expected = last_expected
            last_expected = next_expected

        # start a write
        def write(wr_addr_i, wr_we_i, wr_data_i):
            yield dut.wr_addr_i.eq(wr_addr_i)
            yield dut.wr_we_i.eq(wr_we_i)
            yield dut.wr_data_i.eq(wr_data_i)

        def process():
            # write a pair of values, one for each memory
            yield from read(0)
            yield from write(0x42, 0b1111, 0x87654321)
            yield
            yield from read(0x42, 0x87654321)
            yield from write(0x43, 0b1111, 0x0FEDCBA9)
            yield
            # skip a beat
            yield from read(0x43, 0x0FEDCBA9)
            yield from write(0, 0, 0)
            yield
            # write again, but now they switch memories
            yield from read(0)
            yield from write(0x42, 0b1111, 0x12345678)
            yield
            yield from read(0x42, 0x12345678)
            yield from write(0x43, 0b1111, 0x9ABCDEF0)
            yield
            yield from read(0x43, 0x9ABCDEF0)
            yield from write(0, 0, 0)
            yield
            # test partial writes
            yield from read(0)
            yield from write(0x42, 0b1001, 0x78FFFF12)
            yield
            yield from read(0)
            yield from write(0x43, 0b0110, 0xFFDEABFF)
            yield
            yield from read(0x42, 0x78345612)
            yield from write(0, 0, 0)
            yield
            yield from read(0x43, 0x9ADEABF0)
            yield from write(0, 0, 0)
            yield
            yield from read(0)
            yield from write(0, 0, 0)
            yield
            if transparent:
                # returns the value just written
                yield from read(0x42, 0x78AA5666)
            else:
                # returns the old value
                yield from read(0x42, 0x78345612)
            yield from write(0x42, 0b0101, 0x55AA9966)
            yield
            # after a cycle, always returns the new value
            yield from read(0x42, 0x78AA5666)
            yield from write(0, 0, 0)
            yield
            yield from read(0)
            yield from write(0, 0, 0)
            yield
            yield from read(0)
            yield from write(0, 0, 0)

        sim.add_sync_process(process)
        debug_file = 'test_dual_port_regfile'
        if transparent:
            debug_file += '_transparent'
        traces = ['clk', 'phase',
                  {'comment': 'write port'},
                  'wr_addr_i[6:0]', 'wr_we_i[3:0]', 'wr_data_i[31:0]',
                  {'comment': 'read port'},
                  'rd_addr_i[6:0]', 'rd_data_o[31:0]',
                  {'comment': 'LVT write port'},
                  'phase', 'lvt_mem_w_addr[6:0]', 'lvt_mem_w_en[3:0]',
                  'lvt_mem_w_data[3:0]',
                  {'comment': 'LVT read port'},
                  'lvt_mem_r_addr[6:0]', 'lvt_mem_r_data[3:0]',
                  {'comment': 'backing memory'},
                  'mem0.rd_data_o[31:0]',
                  'mem1.rd_data_o[31:0]',
                  ]
        write_gtkw(debug_file + '.gtkw',
                   debug_file + '.vcd',
                   traces, module='top', zoom=-22)
        sim_writer = sim.write_vcd(debug_file + '.vcd')
        with sim_writer:
            sim.run()

    def test_dual_port_regfile(self):
        with self.subTest("non-transparent reads"):
            self.do_test_dual_port_regfile(False)
        with self.subTest("transparent reads"):
            self.do_test_dual_port_regfile(True)

    def do_test_dual_port_regfile_proof(self, transparent=True):
        """
        Formal proof of the 1W/1R regfile
        """
        m = Module()
        # 128 x 32-bit, 8-bit granularity
        dut = DualPortRegfile(7, 32, 4, transparent)
        m.submodules.dut = dut
        gran = dut.data_width // dut.we_width  # granularity
        # choose a single random memory location to test
        a_const = AnyConst(dut.addr_width)
        # choose a single byte lane to test
        lane = AnyConst(range(dut.we_width))
        # holding data register
        d_reg = Signal(gran)
        # keep track of the phase, so we can remember which memory
        # we wrote to
        phase = Signal()
        m.d.sync += phase.eq(~phase)
        # for some reason, simulated formal memory is not zeroed at reset
        # ... so, remember whether we wrote it, at least once.
        wrote = Signal()
        # ... and on which phase it was written
        wrote_phase = Signal()
        # if our memory location and byte lane is being written,
        # capture the data in our holding register
        with m.If((dut.wr_addr_i == a_const)
                  & dut.wr_we_i.bit_select(lane, 1)):
            m.d.sync += d_reg.eq(dut.wr_data_i.word_select(lane, gran))
            m.d.sync += wrote.eq(1)
            m.d.sync += wrote_phase.eq(phase)
        # if our memory location is being read,
        # and the holding register has valid data,
        # then its value must match the memory output, on the given lane
        with m.If(Past(dut.rd_addr_i) == a_const):
            if transparent:
                with m.If(wrote):
                    rd_lane = dut.rd_data_o.word_select(lane, gran)
                    m.d.sync += Assert(d_reg == rd_lane)
            else:
                # with a non-transparent read port, the read value depends
                # on whether there is a simultaneous write, or not
                with m.If(Past(dut.wr_addr_i) == a_const):
                    # simultaneous write -> check against last written value
                    with m.If(wrote & Past(wrote)):
                        rd_lane = dut.rd_data_o.word_select(lane, gran)
                        m.d.sync += Assert(Past(d_reg) == rd_lane)
                with m.Else():
                    # otherwise, check against current written value
                    with m.If(wrote):
                        rd_lane = dut.rd_data_o.word_select(lane, gran)
                        m.d.sync += Assert(d_reg == rd_lane)

        m.d.comb += [
            dut.dbg_addr.eq(a_const),
            dut.dbg_lane.eq(lane),
            dut.dbg_data.eq(d_reg),
            dut.dbg_wrote.eq(wrote),
            dut.dbg_wrote_phase.eq(wrote_phase),
            dut.dbg_phase.eq(phase),
        ]

        self.assertFormal(m, mode="prove", depth=3)

    def test_dual_port_regfile_proof(self):
        """
        Formal check of 1W/1R regfile (transparent and not)
        """
        with self.subTest("transparent reads"):
            self.do_test_dual_port_regfile_proof(True)
        with self.subTest("non-transparent reads"):
            self.do_test_dual_port_regfile_proof(False)


class PhasedReadPhasedWriteFullReadSRAM(Elaboratable):
    """
    Builds, from three 1RW blocks, a pseudo 1W/2R SRAM, with:

    * one full read port, which works every cycle,
    * one write port, which is only available on either even or odd cycles,
    * an extra transparent read port, available only on the same cycles as the
      write port

    This type of SRAM is useful for a XOR-based 6x1RW implementation of
    a 1R/1W register file.

    :param addr_width: width of the address bus
    :param data_width: width of the data bus
    :param we_width: number of write enable lines
    :param write_phase: indicates on which phase the write port will
                        accept data
    :param transparent: whether a simultaneous read and write returns the
                        new value (True) or the old value (False) on the full
                        read port

    .. note:: The debug read port is meant only to assist in formal proofs!
    """

    def __init__(self, addr_width, data_width, we_width, write_phase,
                 transparent=True):
        self.addr_width = addr_width
        self.data_width = data_width
        self.we_width = we_width
        self.write_phase = write_phase
        self.transparent = transparent
        # interface signals
        self.wr_addr_i = Signal(addr_width); """phased write port address"""
        self.wr_data_i = Signal(data_width); """phased write port data"""
        self.wr_we_i = Signal(we_width); """phased write port enable"""
        self.rd_addr_i = Signal(addr_width); """full read port address"""
        self.rd_data_o = Signal(data_width); """full read port data"""
        self.rdp_addr_i = Signal(addr_width); """phased read port address"""
        self.rdp_data_o = Signal(data_width); """phased read port data"""
        self.phase = Signal(); """even/odd cycle indicator"""
        # debug signals, only used in formal proofs
        self.dbg_addr = Signal(addr_width); """debug: address under test"""
        lanes = range(we_width)
        self.dbg_lane = Signal(lanes); """debug: write lane under test"""
        gran = self.data_width // self.we_width
        self.dbg_data = Signal(gran); """debug: data to keep in sync"""
        self.dbg_wrote = Signal(); """debug: data is valid"""

    def elaborate(self, platform):
        m = Module()
        # instantiate the 1RW memory blocks
        mem1 = SinglePortSRAM(self.addr_width, self.data_width, self.we_width)
        mem2 = SinglePortSRAM(self.addr_width, self.data_width, self.we_width)
        mem3 = SinglePortSRAM(self.addr_width, self.data_width, self.we_width)
        m.submodules.mem1 = mem1
        m.submodules.mem2 = mem2
        m.submodules.mem3 = mem3
        # wire input write data to first memory, and its output to the others
        m.d.comb += [
            mem1.d.eq(self.wr_data_i),
            mem2.d.eq(mem1.q),
            mem3.d.eq(mem1.q)
        ]
        # holding registers for the write port of the other memories
        last_wr_addr = Signal(self.addr_width)
        last_wr_we = Signal(self.we_width)
        # do read and write addresses coincide?
        same_read_write = Signal()
        same_phased_read_write = Signal()
        with m.If(self.phase == self.write_phase):
            # write phase, start a write on the first memory
            m.d.comb += mem1.a.eq(self.wr_addr_i)
            m.d.comb += mem1.we.eq(self.wr_we_i)
            # save write address and write select for repeating the write
            # on the other memories, one cycle later
            m.d.sync += last_wr_we.eq(self.wr_we_i)
            m.d.sync += last_wr_addr.eq(self.wr_addr_i)
            # start a read on the other memories
            m.d.comb += mem2.a.eq(self.rd_addr_i)
            m.d.comb += mem3.a.eq(self.rdp_addr_i)
            # output previously read data from the first memory
            m.d.comb += self.rd_data_o.eq(mem1.q)
            # remember whether we are reading from the same location as we
            # are writing
            m.d.sync += same_phased_read_write.eq(
                self.rdp_addr_i == self.wr_addr_i)
            if self.transparent:
                m.d.sync += same_read_write.eq(self.rd_addr_i == self.wr_addr_i)
        with m.Else():
            # read phase, write last written data on the other memories
            m.d.comb += [
                mem2.a.eq(last_wr_addr),
                mem2.we.eq(last_wr_we),
                mem3.a.eq(last_wr_addr),
                mem3.we.eq(last_wr_we),
            ]
            # start a read on the first memory
            m.d.comb += mem1.a.eq(self.rd_addr_i)
            # output the read data from the second memory
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
            with m.If(same_phased_read_write):
                # if read and write addresses coincide,
                # output the data just written
                m.d.comb += self.rdp_data_o.eq(mem1.q)
            with m.Else():
                # otherwise, output previously read data
                # from the third memory
                m.d.comb += self.rdp_data_o.eq(mem3.q)

        if platform == "formal":
            # pass our state to the device under test, so it can ensure that
            # its state is in sync with ours, for induction
            m.d.comb += [
                # pass the address and write lane under test to both memories
                mem1.dbg_addr.eq(self.dbg_addr),
                mem2.dbg_addr.eq(self.dbg_addr),
                mem3.dbg_addr.eq(self.dbg_addr),
                mem1.dbg_lane.eq(self.dbg_lane),
                mem2.dbg_lane.eq(self.dbg_lane),
                mem3.dbg_lane.eq(self.dbg_lane),
                # the other memories copy their state from the first memory,
                # after a cycle, so they have a one cycle delay
                mem1.dbg_data.eq(self.dbg_data),
                mem2.dbg_data.eq(Past(self.dbg_data)),
                mem3.dbg_data.eq(Past(self.dbg_data)),
                mem1.dbg_wrote.eq(self.dbg_wrote),
                mem2.dbg_wrote.eq(Past(self.dbg_wrote)),
                mem3.dbg_wrote.eq(Past(self.dbg_wrote)),
            ]

        return m


class PhasedReadPhasedWriteFullReadSRAMTestCase(FHDLTestCase):

    def do_test_case(self, write_phase, transparent):
        """
        Simulate some read/write/modify operations
        """
        dut = PhasedReadPhasedWriteFullReadSRAM(7, 32, 4, write_phase,
                                                transparent)
        sim = Simulator(dut)
        sim.add_clock(1e-6)

        expected = None
        last_expected = None

        # compare read data with previously written data
        # and start a new read
        def read(rd_addr_i, next_expected=None):
            nonlocal expected, last_expected
            if expected is not None:
                self.assertEqual((yield dut.rd_data_o), expected)
            yield dut.rd_addr_i.eq(rd_addr_i)
            # account for the read latency
            expected = last_expected
            last_expected = next_expected

        expected2 = None

        # same as above, but for the phased read port
        def phased_read(rdp_addr_i, next_expected2=None):
            nonlocal expected2
            if expected2 is not None:
                self.assertEqual((yield dut.rdp_data_o), expected2)
            yield dut.rdp_addr_i.eq(rdp_addr_i)
            # account for the read latency
            expected2 = next_expected2

        # start a write
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
            # also skip reading from the phased read port
            yield dut.rdp_addr_i.eq(0)

        # writes a few values on the write port, and read them back
        def process():
            yield from read(0)
            yield from phased_read(0)
            yield from write(0x42, 0b1111, 0x12345678)
            yield
            yield from read(0x42, 0x12345678)
            yield from skip_write()
            yield
            yield from read(0x42, 0x12345678)
            yield from phased_read(0x42, 0x12345678)
            yield from write(0x43, 0b1111, 0x9ABCDEF0)
            yield
            yield from read(0x43, 0x9ABCDEF0)
            yield from skip_write()
            yield
            yield from read(0x42, 0x12345678)
            yield from phased_read(0x42, 0x12345678)
            yield from write(0x43, 0b1001, 0xF0FFFF9A)
            yield
            yield from read(0x43, 0xF0BCDE9A)
            yield from skip_write()
            yield
            yield from read(0x43, 0xF0BCDE9A)
            yield from phased_read(0x43, 0xF0BCDE9A)
            yield from write(0x42, 0b0110, 0xFF5634FF)
            yield
            yield from read(0x42, 0x12563478)
            yield from skip_write()
            yield
            yield from read(0)
            yield from phased_read(0)
            yield from write(0, 0, 0)
            yield
            yield from read(0)
            yield from skip_write()
            yield
            # try reading and writing at the same time
            if transparent:
                # transparent port, return the value just written
                yield from read(0x42, 0x12AA3466)
            else:
                # ... otherwise, return the old value
                yield from read(0x42, 0x12563478)
            # transparent port, always return the value just written
            yield from phased_read(0x42, 0x12AA3466)
            yield from write(0x42, 0b0101, 0x55AA9966)
            yield
            # after a cycle, always returns the new value
            yield from read(0x42, 0x12AA3466)
            yield from skip_write()
            yield
            yield from read(0)
            yield from phased_read(0)
            yield from write(0, 0, 0)
            yield
            yield from read(0)
            yield from skip_write()

        sim.add_sync_process(process)
        debug_file = 'test_phased_read_write_sram_' + str(write_phase)
        if transparent:
            debug_file += '_transparent'
        traces = ['clk', 'phase',
                  {'comment': 'phased write port'},
                  'wr_addr_i[6:0]', 'wr_we_i[3:0]', 'wr_data_i[31:0]',
                  {'comment': 'full read port'},
                  'rd_addr_i[6:0]', 'rd_data_o[31:0]',
                  {'comment': 'phased read port'},
                  'rdp_addr_i[6:0]', 'rdp_data_o[31:0]']
        write_gtkw(debug_file + '.gtkw',
                   debug_file + '.vcd',
                   traces, module='top', zoom=-22)
        sim_writer = sim.write_vcd(debug_file + '.vcd')
        with sim_writer:
            sim.run()

    def test_case(self):
        """test both types (odd and even write ports) of phased memory"""
        with self.subTest("writes happen on phase 0"):
            self.do_test_case(0, True)
        with self.subTest("writes happen on phase 1"):
            self.do_test_case(1, True)
        with self.subTest("writes happen on phase 0 (non-transparent reads)"):
            self.do_test_case(0, False)
        with self.subTest("writes happen on phase 1 (non-transparent reads)"):
            self.do_test_case(1, False)

    def do_test_formal(self, write_phase, transparent):
        """
        Formal proof of the pseudo 1W/2R regfile
        """
        m = Module()
        # 128 x 32-bit, 8-bit granularity
        dut = PhasedReadPhasedWriteFullReadSRAM(7, 32, 4, write_phase,
                                                transparent)
        m.submodules.dut = dut
        gran = dut.data_width // dut.we_width  # granularity
        # choose a single random memory location to test
        a_const = AnyConst(dut.addr_width)
        # choose a single byte lane to test
        lane = AnyConst(range(dut.we_width))
        # drive alternating phases
        m.d.comb += Assume(dut.phase != Past(dut.phase))
        # holding data register
        d_reg = Signal(gran)
        # for some reason, simulated formal memory is not zeroed at reset
        # ... so, remember whether we wrote it, at least once.
        wrote = Signal()
        # if our memory location and byte lane is being written,
        # capture the data in our holding register
        with m.If((dut.wr_addr_i == a_const)
                  & dut.wr_we_i.bit_select(lane, 1)
                  & (dut.phase == dut.write_phase)):
            m.d.sync += d_reg.eq(dut.wr_data_i.word_select(lane, gran))
            m.d.sync += wrote.eq(1)
        # if our memory location is being read,
        # and the holding register has valid data,
        # then its value must match the memory output, on the given lane
        with m.If(Past(dut.rd_addr_i) == a_const):
            if transparent:
                with m.If(wrote):
                    rd_lane = dut.rd_data_o.word_select(lane, gran)
                    m.d.sync += Assert(d_reg == rd_lane)
            else:
                # with a non-transparent read port, the read value depends
                # on whether there is a simultaneous write, or not
                with m.If((Past(dut.wr_addr_i) == a_const)
                          & Past(dut.phase) == dut.write_phase):
                    # simultaneous write -> check against last written value
                    with m.If(Past(wrote)):
                        rd_lane = dut.rd_data_o.word_select(lane, gran)
                        m.d.sync += Assert(Past(d_reg) == rd_lane)
                with m.Else():
                    # otherwise, check against current written value
                    with m.If(wrote):
                        rd_lane = dut.rd_data_o.word_select(lane, gran)
                        m.d.sync += Assert(d_reg == rd_lane)
        # same for the phased read port, except it's always transparent
        # and the port works only on the write phase
        with m.If((Past(dut.rdp_addr_i) == a_const) & wrote
                  & (Past(dut.phase) == dut.write_phase)):
            rdp_lane = dut.rdp_data_o.word_select(lane, gran)
            m.d.sync += Assert(d_reg == rdp_lane)

        # pass our state to the device under test, so it can ensure that
        # its state is in sync with ours, for induction
        m.d.comb += [
            # address and mask under test
            dut.dbg_addr.eq(a_const),
            dut.dbg_lane.eq(lane),
            # state of our holding register
            dut.dbg_data.eq(d_reg),
            dut.dbg_wrote.eq(wrote),
        ]

        self.assertFormal(m, mode="prove", depth=3)

    def test_formal(self):
        """test both types (odd and even write ports) of phased write memory"""
        with self.subTest("writes happen on phase 0"):
            self.do_test_formal(0, False)
        with self.subTest("writes happen on phase 1"):
            self.do_test_formal(1, False)
        # test again, with transparent read ports
        with self.subTest("writes happen on phase 0 (transparent reads)"):
            self.do_test_formal(0, True)
        with self.subTest("writes happen on phase 1 (transparent reads)"):
            self.do_test_formal(1, True)


class DualPortXorRegfile(Elaboratable):
    """
    Builds, from a pair of phased 1W/2R blocks, a true 1W/1R RAM, where both
    write and (non-transparent) read ports work every cycle.

    It employs a XOR trick, as follows:

    1) Like before, there are two memories, each reading on every cycle, and
       writing on alternate cycles
    2) Instead of a MUX, the read port is a direct XOR of the two memories.
    3) Writes happens in two cycles:

        First, read the current value of the *other* memory, at the write
        location.

        Then, on *this* memory, write that read value, XORed with the desired
        value.

    This recovers the desired value when read:
    (other XOR desired) XOR other = desired

    :param addr_width: width of the address bus
    :param data_width: width of the data bus
    :param we_width: number of write enable lines
    """

    def __init__(self, addr_width, data_width, we_width):
        self.addr_width = addr_width
        self.data_width = data_width
        self.we_width = we_width
        # interface signals
        self.wr_addr_i = Signal(addr_width); """write port address"""
        self.wr_data_i = Signal(data_width); """write port data"""
        self.wr_we_i = Signal(we_width); """write port enable"""
        self.rd_addr_i = Signal(addr_width); """read port address"""
        self.rd_data_o = Signal(data_width); """read port data"""

    def elaborate(self, platform):
        m = Module()
        # instantiate the two phased 1W/2R memory blocks
        mem0 = PhasedReadPhasedWriteFullReadSRAM(
            self.addr_width, self.data_width, self.we_width, 0, True)
        mem1 = PhasedReadPhasedWriteFullReadSRAM(
            self.addr_width, self.data_width, self.we_width, 1, True)
        m.submodules.mem0 = mem0
        m.submodules.mem1 = mem1
        # generate and wire the phases for the phased memories
        phase = Signal()
        m.d.sync += phase.eq(~phase)
        m.d.comb += [
            mem0.phase.eq(phase),
            mem1.phase.eq(phase),
        ]
        # wire read address to memories, and XOR their output
        m.d.comb += [
            mem0.rd_addr_i.eq(self.rd_addr_i),
            mem1.rd_addr_i.eq(self.rd_addr_i),
            self.rd_data_o.eq(mem0.rd_data_o ^ mem1.rd_data_o),
        ]
        # write path
        # 1) read the memory location which is about to be written
        m.d.comb += [
            mem0.rdp_addr_i.eq(self.wr_addr_i),
            mem1.rdp_addr_i.eq(self.wr_addr_i),
        ]
        # store the write information for the next cycle
        last_addr = Signal(self.addr_width)
        last_we = Signal(self.we_width)
        last_data = Signal(self.data_width)
        m.d.sync += [
            last_addr.eq(self.wr_addr_i),
            last_we.eq(self.wr_we_i),
            last_data.eq(self.wr_data_i),
        ]
        # 2) write the XOR of the other memory data, and the desired value
        m.d.comb += [
            mem0.wr_addr_i.eq(last_addr),
            mem1.wr_addr_i.eq(last_addr),
            mem0.wr_we_i.eq(last_we),
            mem1.wr_we_i.eq(last_we),
            mem0.wr_data_i.eq(last_data ^ mem1.rdp_data_o),
            mem1.wr_data_i.eq(last_data ^ mem0.rdp_data_o),
        ]
        return m


class DualPortXorRegfileTestCase(FHDLTestCase):

    def test_case(self):
        """
        Simulate some read/write/modify operations on the dual port register
        file
        """
        dut = DualPortXorRegfile(7, 32, 4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)

        expected = None
        last_expected = None

        # compare read data with previously written data
        # and start a new read
        def read(rd_addr_i, next_expected=None):
            nonlocal expected, last_expected
            if expected is not None:
                self.assertEqual((yield dut.rd_data_o), expected)
            yield dut.rd_addr_i.eq(rd_addr_i)
            # account for the read latency
            expected = last_expected
            last_expected = next_expected

        # start a write
        def write(wr_addr_i, wr_we_i, wr_data_i):
            yield dut.wr_addr_i.eq(wr_addr_i)
            yield dut.wr_we_i.eq(wr_we_i)
            yield dut.wr_data_i.eq(wr_data_i)

        def process():
            # write a pair of values, one for each memory
            yield from read(0)
            yield from write(0x42, 0b1111, 0x87654321)
            yield
            yield from read(0x42, 0x87654321)
            yield from write(0x43, 0b1111, 0x0FEDCBA9)
            yield
            # skip a beat
            yield from read(0x43, 0x0FEDCBA9)
            yield from write(0, 0, 0)
            yield
            # write again, but now they switch memories
            yield from read(0)
            yield from write(0x42, 0b1111, 0x12345678)
            yield
            yield from read(0x42, 0x12345678)
            yield from write(0x43, 0b1111, 0x9ABCDEF0)
            yield
            yield from read(0x43, 0x9ABCDEF0)
            yield from write(0, 0, 0)
            yield
            # test partial writes
            yield from read(0)
            yield from write(0x42, 0b1001, 0x78FFFF12)
            yield
            yield from read(0)
            yield from write(0x43, 0b0110, 0xFFDEABFF)
            yield
            yield from read(0x42, 0x78345612)
            yield from write(0, 0, 0)
            yield
            yield from read(0x43, 0x9ADEABF0)
            yield from write(0, 0, 0)
            yield
            yield from read(0)
            yield from write(0, 0, 0)
            yield
            # test simultaneous read and write
            # non-transparent read: returns the old value
            yield from read(0x42, 0x78345612)
            yield from write(0x42, 0b0101, 0x55AA9966)
            yield
            # after a cycle, returns the new value
            yield from read(0x42, 0x78AA5666)
            yield from write(0, 0, 0)
            yield
            # settle down
            yield from read(0)
            yield from write(0, 0, 0)
            yield
            yield from read(0)
            yield from write(0, 0, 0)

        sim.add_sync_process(process)
        debug_file = 'test_dual_port_xor_regfile'
        traces = ['clk', 'phase',
                  {'comment': 'write port'},
                  'wr_addr_i[6:0]', 'wr_we_i[3:0]', 'wr_data_i[31:0]',
                  {'comment': 'read port'},
                  'rd_addr_i[6:0]', 'rd_data_o[31:0]',
                  ]
        write_gtkw(debug_file + '.gtkw',
                   debug_file + '.vcd',
                   traces, module='top', zoom=-22)
        sim_writer = sim.write_vcd(debug_file + '.vcd')
        with sim_writer:
            sim.run()


if __name__ == "__main__":
    unittest.main()
