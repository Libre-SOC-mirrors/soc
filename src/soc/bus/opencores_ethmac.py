#!/usr/bin/env python3
#
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2020-2022 Raptor Engineering LLC <support@raptorengineering.com>
# Copyright (C) 2022 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Sponsored by NLnet and NGI POINTER under EU Grants 871528 and 957073
# Part of the Libre-SOC Project.
#
# this is a wrapper around the opencores verilog 10/100 MAC

from nmigen import (Elaboratable, Cat, Module, Signal, ClockSignal, Instance,
                    ResetSignal)

from nmigen_soc.wishbone.bus import Interface
from nmigen_soc.memory import MemoryMap
from lambdasoc.periph.event import IRQLine
from nmigen.utils import log2_int
from nmigen.cli import rtlil, verilog
import os

__all__ = ["EthMAC"]


class EthMAC(Elaboratable):
    """Ethernet MAC from opencores, nmigen wrapper.
    remember to call EthMAC.add_verilog_source
    """

    def __init__(self, master_bus=None, slave_bus=None, name=None,
                       irq=None, pins=None):
        if name is not None:
            # convention: give the name in the format "name_number"
            self.idx = int(name.split("_")[-1])
        else:
            self.idx = 0
            name = "eth_0"
        self.granularity = 8
        self.data_width = 32
        self.dsize = log2_int(self.data_width//self.granularity)

        # set up the wishbone busses
        features = frozenset()
        if master_bus is None:
            master_bus = Interface(addr_width=30,
                            data_width=32,
                            features=features,
                            granularity=8,
                            name=name+"_wb_%d_0" % self.idx)
        if slave_bus is None:
            slave_bus = Interface(addr_width=12,
                            data_width=32,
                            features=features,
                            granularity=8,
                            name=name+"_wb_%d_1" % self.idx)
        self.master_bus = master_bus
        self.slave_bus = slave_bus
        if irq is None:
            irq = IRQLine()
        self.irq = irq

        slave_mmap = MemoryMap(addr_width=12+self.dsize,
                        data_width=self.granularity)

        self.slave_bus.memory_map = slave_mmap

        # RMII TX signals
        self.mtx_clk = Signal()
        self.mtxd = Signal(4)
        self.mtxen = Signal()
        self.mtxerr = Signal()

        # RMII RX signals
        self.mrx_clk = Signal()
        self.mrxd = Signal(4)
        self.mrxdv = Signal()
        self.mrxerr = Signal()

        # RMII common signals
        self.mcoll = Signal()
        self.mcrs = Signal()

        # RMII management interface signals
        self.mdc = Signal()
        self.md_in = Signal()
        self.md_out = Signal()
        self.md_direction = Signal()

        # pins resource
        self.pins = pins

    @classmethod
    def add_verilog_source(cls, verilog_src_dir, platform):
        # add each of the verilog sources, needed for when doing platform.build
        for fname in ['eth_clockgen.v', 'eth_cop.v', 'eth_crc.v',
                    'eth_fifo.v', 'eth_maccontrol.v', 'ethmac_defines.v',
                    'eth_macstatus.v', 'ethmac.v', 'eth_miim.v',
                    'eth_outputcontrol.v', 'eth_random.v',
                    'eth_receivecontrol.v', 'eth_registers.v',
                    'eth_register.v', 'eth_rxaddrcheck.v',
                    'eth_rxcounters.v', 'eth_rxethmac.v',
                    'eth_rxstatem.v', 'eth_shiftreg.v',
                    'eth_spram_256x32.v', 'eth_top.v',
                    'eth_transmitcontrol.v', 'eth_txcounters.v',
                    'eth_txethmac.v', 'eth_txstatem.v', 'eth_wishbone.v',
                    'timescale.v']:
            # prepend the src directory to each filename, add its contents
            fullname = os.path.join(verilog_src_dir, fname)
            with open(fullname) as f:
                platform.add_file(fullname, f)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        idx = self.idx

        # Calculate arbiter bus address
        wb_master_bus_adr = Signal(32)
        # arbiter address is in words, ethernet master address is in bytes
        comb += self.master_bus.adr.eq(wb_master_bus_adr >> 2)

        # create definition of external verilog EthMAC code here, so that
        # nmigen understands I/O directions (defined by i_ and o_ prefixes)
        ethmac = Instance("eth_top",
                            # Clock/reset (use DomainRenamer if needed)
                            i_wb_clk_i=ClockSignal(),
                            i_wb_rst_i=ResetSignal(),

                            # Master Wishbone bus signals
                            o_m_wb_adr_o=wb_master_bus_adr,
                            i_m_wb_dat_i=self.master_bus.dat_r,
                            o_m_wb_sel_o=self.master_bus.sel,
                            o_m_wb_dat_o=self.master_bus.dat_w,
                            o_m_wb_we_o=self.master_bus.we,
                            o_m_wb_stb_o=self.master_bus.stb,
                            o_m_wb_cyc_o=self.master_bus.cyc,
                            i_m_wb_ack_i=self.master_bus.ack,

                            # Slave Wishbone bus signals
                            i_wb_adr_i=self.slave_bus.adr,
                            i_wb_dat_i=self.slave_bus.dat_w,
                            i_wb_sel_i=self.slave_bus.sel,
                            o_wb_dat_o=self.slave_bus.dat_r,
                            i_wb_we_i=self.slave_bus.we,
                            i_wb_stb_i=self.slave_bus.stb,
                            i_wb_cyc_i=self.slave_bus.cyc,
                            o_wb_ack_o=self.slave_bus.ack,

                            o_int_o=self.irq,

                            # RMII TX
                            i_mtx_clk_pad_i=self.mtx_clk,
                            o_mtxd_pad_o=self.mtxd,
                            o_mtxen_pad_o=self.mtxen,
                            o_mtxerr_pad_o=self.mtxerr,

                            # RMII RX
                            i_mrx_clk_pad_i=self.mrx_clk,
                            i_mrxd_pad_i=self.mrxd,
                            i_mrxdv_pad_i=self.mrxdv,
                            i_mrxerr_pad_i=self.mrxerr,

                            # RMII common
                            i_mcoll_pad_i=self.mcoll,
                            i_mcrs_pad_i=self.mcrs,

                            # Management Interface
                            o_mdc_pad_o=self.mdc,
                            i_md_pad_i=self.md_in,
                            o_md_pad_o=self.md_out,
                            o_md_padoe_o=self.md_direction
                            );

        m.submodules['ethmac_%d' % self.idx] = ethmac

        if self.pins is not None:
            comb += self.mtx_clk.eq(self.pins.mtx_clk.i)
            comb += self.pins.mtxd.o.eq(self.mtxd)
            comb += self.pins.mtxen.o.eq(self.mtxen)
            comb += self.pins.mtxerr.o.eq(self.mtxerr)

            comb += self.mrx_clk.eq(self.pins.mrx_clk.i)
            comb += self.mrxd.eq(self.pins.mrxd.i)
            comb += self.mrxdv.eq(self.pins.mrxdv.i)
            comb += self.mrxerr.eq(self.pins.mrxerr.i)
            comb += self.mcoll.eq(self.pins.mcoll.i)
            comb += self.mcrs.eq(self.pins.mcrs.i)

            comb += self.pins.mdc.o.eq(self.mdc)

            comb += self.pins.md.o.eq(self.md_out)
            comb += self.pins.md.oe.eq(self.md_direction)
            comb += self.md_in.eq(self.pins.md.i)
        return m


def create_ilang(dut, ports, test_name):
    vl = rtlil.convert(dut, name=test_name, ports=ports)
    with open("%s.il" % test_name, "w") as f:
        f.write(vl)

def create_verilog(dut, ports, test_name):
    vl = verilog.convert(dut, name=test_name, ports=ports)
    with open("%s.v" % test_name, "w") as f:
        f.write(vl)

if __name__ == "__main__":
    ethmac = EthMAC(name="eth_0")
    create_ilang(ethmac, [ethmac.master_bus.cyc, ethmac.master_bus.stb,
                        ethmac.master_bus.ack, ethmac.master_bus.dat_r,
                        ethmac.master_bus.dat_w, ethmac.master_bus.adr,
                        ethmac.master_bus.we, ethmac.master_bus.sel,
                        ethmac.slave_bus.cyc, ethmac.slave_bus.stb,
                        ethmac.slave_bus.ack,
                        ethmac.slave_bus.dat_r, ethmac.slave_bus.dat_w,
                        ethmac.slave_bus.adr,
                        ethmac.slave_bus.we, ethmac.slave_bus.sel,
                        ethmac.mtx_clk, ethmac.mtxd, ethmac.mtxen,
                        ethmac.mtxerr, ethmac.mrx_clk, ethmac.mrxd,
                        ethmac.mrxdv, ethmac.mrxerr, ethmac.mcoll,
                        ethmac.mcrs, ethmac.mdc, ethmac.md_in,
                        ethmac.md_out, ethmac.md_direction
                       ], "eth_0")

