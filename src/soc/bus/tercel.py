#!/usr/bin/env python3
#
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2020-2022 Raptor Engineering LLC <support@raptorengineering.com>
# Copyright (C) 2022 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Sponsored by NLnet and NGI POINTER under EU Grants 871528 and 957073
# Part of the Libre-SOC Project.
#
# this is a wrapper around the opencores verilog tercel module

from nmigen import (Elaboratable, Cat, Module, Signal, ClockSignal, Instance,
                    ResetSignal, Const)

from nmigen_soc.wishbone.bus import Interface
from nmigen_soc.memory import MemoryMap
from nmigen.utils import log2_int
from nmigen.cli import rtlil, verilog
from nmutil.byterev import byte_reverse
import os

__all__ = ["Tercel"]


class Tercel(Elaboratable):
    """Tercel SPI controller from Raptor Engineering, nmigen wrapper.
    remember to call Tercel.add_verilog_source
    """

    def __init__(self, bus=None, cfg_bus=None, features=None, name=None,
                       data_width=32, spi_region_addr_width=28, pins=None,
                       clk_freq=None,
                       lattice_ecp5_usrmclk=False,
                       adr_offset=0): # address offset (bytes)
        if name is not None:
            # convention: give the name in the format "name_number"
            self.idx = int(name.split("_")[-1])
        else:
            self.idx = 0
            name = "spi_0"
        self.granularity = 8
        self.data_width = data_width
        self.dsize = log2_int(self.data_width//self.granularity)
        self.adr_offset = adr_offset
        self.lattice_ecp5_usrmclk = lattice_ecp5_usrmclk

        # TODO, sort this out.
        assert clk_freq is not None
        clk_freq = round(clk_freq)
        self.clk_freq = Const(clk_freq, 32) #clk_freq.bit_length())

        # set up the wishbone busses
        if features is None:
            features = frozenset({'err'})
        if bus is None:
            bus = Interface(addr_width=spi_region_addr_width,
                            data_width=data_width,
                            features=features,
                            granularity=8,
                            name=name+"_wb_%d_0" % self.idx)
        if cfg_bus is None:
            cfg_bus = Interface(addr_width=6,
                            data_width=data_width,
                            features=features,
                            granularity=8,
                            name=name+"_wb_%d_1" % self.idx)
        self.bus = bus
        assert len(self.bus.dat_r) == data_width, \
                        "bus width must be %d" % data_width
        self.cfg_bus = cfg_bus
        assert len(self.cfg_bus.dat_r) == data_width, \
                        "bus width must be %d" % data_width

        mmap = MemoryMap(addr_width=spi_region_addr_width+self.dsize,
                        data_width=self.granularity)
        cfg_mmap = MemoryMap(addr_width=6+self.dsize,
                        data_width=self.granularity)

        self.bus.memory_map = mmap
        self.cfg_bus.memory_map = cfg_mmap

        # QSPI signals
        self.dq_out = Signal(4)       # Data
        self.dq_direction = Signal(4)
        self.dq_in = Signal(4)
        self.cs_n_out = Signal()      # Slave select
        self.spi_clk = Signal()       # Clock
        self.dbg_port = Signal(8)     # debug info

        # pins resource
        self.pins = pins

    @classmethod
    def add_verilog_source(cls, verilog_src_dir, platform):
        # add each of the verilog sources, needed for when doing platform.build
        for fname in ['wishbone_spi_master.v', 'phy.v']:
            # prepend the src directory to each filename, add its contents
            fullname = os.path.join(verilog_src_dir, fname)
            with open(fullname) as f:
                platform.add_file(fullname, f)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        pins, bus, cfg_bus = self.pins, self.bus, self.cfg_bus

        # Calculate SPI flash address
        spi_bus_adr = Signal(30)
        # wb address is in words, offset is in bytes
        comb += spi_bus_adr.eq(bus.adr - (self.adr_offset >> 2))

        # urrr.... byte-reverse the config bus and data bus read/write
        cdat_w = Signal.like(cfg_bus.dat_w)
        cdat_r = Signal.like(cfg_bus.dat_r)
        dat_w = Signal.like(bus.dat_w)
        dat_r = Signal.like(bus.dat_r)
        comb += cdat_w.eq(byte_reverse(m, "rv_cdat_w", cfg_bus.dat_w, 4))
        comb += cfg_bus.dat_r.eq(byte_reverse(m, "rv_cdat_r", cdat_r, 4))
        comb += dat_w.eq(byte_reverse(m, "rv_dat_w", bus.dat_w, 4))
        comb += bus.dat_r.eq(byte_reverse(m, "rv_dat_r", dat_r, 4))

        # create definition of external verilog Tercel code here, so that
        # nmigen understands I/O directions (defined by i_ and o_ prefixes)
        idx, bus = self.idx, self.bus
        tercel = Instance("tercel_core",
                            # System parameters
                            i_sys_clk_freq = self.clk_freq,

                            # Clock/reset (use DomainRenamer if needed)
                            i_peripheral_clock=ClockSignal(),
                            i_peripheral_reset=ResetSignal(),

                            # SPI region Wishbone bus signals
                            i_wishbone_adr=spi_bus_adr,
                            i_wishbone_dat_w=dat_w,
                            i_wishbone_sel=bus.sel,
                            o_wishbone_dat_r=dat_r,
                            i_wishbone_we=bus.we,
                            i_wishbone_stb=bus.stb,
                            i_wishbone_cyc=bus.cyc,
                            o_wishbone_ack=bus.ack,
                            o_wishbone_err=bus.err,

                            # Configuration region Wishbone bus signals
                            i_cfg_wishbone_adr=cfg_bus.adr,
                            i_cfg_wishbone_dat_w=cdat_w,
                            i_cfg_wishbone_sel=cfg_bus.sel,
                            o_cfg_wishbone_dat_r=cdat_r,
                            i_cfg_wishbone_we=cfg_bus.we,
                            i_cfg_wishbone_stb=cfg_bus.stb,
                            i_cfg_wishbone_cyc=cfg_bus.cyc,
                            o_cfg_wishbone_ack=cfg_bus.ack,
                            o_cfg_wishbone_err=cfg_bus.err,

                            # QSPI signals
                            o_spi_d_out=self.dq_out,
                            o_spi_d_direction=self.dq_direction,
                            i_spi_d_in=self.dq_in,
                            o_spi_ss_n=self.cs_n_out,
                            o_spi_clock=self.spi_clk,

                            # debug port
                            o_debug_port=self.dbg_port
                            );

        m.submodules['tercel_%d' % self.idx] = tercel

        if pins is not None:
            comb += pins.dq.o.eq(self.dq_out)
            comb += pins.dq.oe.eq(self.dq_direction)
            comb += pins.dq.oe.eq(self.dq_direction)
            comb += pins.dq.o_clk.eq(ClockSignal())
            comb += self.dq_in.eq(pins.dq.i)
            comb += pins.dq.i_clk.eq(ClockSignal())
            # XXX invert handled by SPIFlashResource
            comb += pins.cs.eq(~self.cs_n_out)
            # ECP5 needs special handling for the SPI clock, sigh.
            if self.lattice_ecp5_usrmclk:
                m.submodules += Instance("USRMCLK",
                    i_USRMCLKI  = self.spi_clk,
                    i_USRMCLKTS = 0
                )
            else:
                comb += pins.clk.eq(self.spi_clk)

        return m

    def ports(self):
        return [self.bus.cyc, self.bus.stb, self.bus.ack,
                        self.bus.dat_r, self.bus.dat_w, self.bus.adr,
                        self.bus.we, self.bus.sel,
                        self.cfg_bus.cyc, self.cfg_bus.stb,
                        self.cfg_bus.ack,
                        self.cfg_bus.dat_r, self.cfg_bus.dat_w,
                        self.cfg_bus.adr,
                        self.cfg_bus.we, self.cfg_bus.sel,
                        self.dq_out, self.dq_direction, self.dq_in,
                        self.cs_n_out, self.spi_clk
                       ]


def create_ilang(dut, ports, test_name):
    vl = rtlil.convert(dut, name=test_name, ports=ports)
    with open("%s.il" % test_name, "w") as f:
        f.write(vl)

def create_verilog(dut, ports, test_name):
    vl = verilog.convert(dut, name=test_name, ports=ports)
    with open("%s.v" % test_name, "w") as f:
        f.write(vl)


if __name__ == "__main__":
    tercel = Tercel(name="spi_0", data_width=32, clk_freq=100e6)
    create_ilang(tercel, tercel.ports(), "spi_0")

