#!/usr/bin/env python3
#
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2022 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Sponsored by NLnet and NGI POINTER under EU Grants 871528 and 957073
# Part of the Libre-SOC Project.
#
# this is a wrapper around the opencores verilog uart16550 module

from nmigen import (Elaboratable, Cat, Module, Signal, ClockSignal, Instance,
                    ResetSignal)

from nmigen_soc.wishbone.bus import Interface
from nmigen.cli import rtlil, verilog
import os

__all__ = ["SDRAM"]


class SDRAM(Elaboratable):
    """SDRAM controller from opencores, nmigen wrapper.  remember to call
       SDRAM.add_verilog_source.

    * the SDRAM IC will be accessible over the Wishbone Bus
    * sdr_* signals must be wired to the IC
    * cfg_* parameters must match those listed in the SDRAM IC's datasheet
    """

    def __init__(self, bus=None, features=None, name=None,
                       data_width=32, addr_width=26,
                       sdr_data_width=16,
                       pins=None):
        if name is not None:
            name = "sdram"
        self.data_width = data_width
        self.sdr_data_width = sdr_data_width
        self.addr_width = addr_width
        self.refresh_timer_sz = 12
        self.refresh_row_count = 3

        # set up the wishbone bus
        if features is None:
            features = frozenset({'cti'})
        if bus is None:
            bus = Interface(addr_width=addr_width,
                            data_width=data_width,
                            features=features,
                            granularity=8,
                            name=name)
        self.bus = bus
        assert len(self.bus.dat_r) == data_width, \
                        "bus width must be %d" % data_width

        byte_width = sdr_data_width // 8 # for individual byte masks/enables

        # SDRAM signals
        self.sdram_clk     = Signal()           # sdram phy clock
        self.sdram_resetn  = Signal(reset_less=True) # sdram reset (low)
        self.sdr_cs_n      = Signal()           # chip select
        self.sdr_cke       = Signal()           # clock-enable
        self.sdr_ras_n     = Signal()           # read-address strobe
        self.sdr_cas_n     = Signal()           # cas
        self.sdr_we_n      = Signal()           # write-enable
        self.sdr_dqm       = Signal(byte_width) # data mask
        self.sdr_ba        = Signal(2)          # bank enable
        self.sdr_addr      = Signal(13)         # sdram address, 13 bits
        # these combine to create a bi-direction inout, sdr_dq
        # note, each bit of sdr_den_n covers a *byte* of sdr_din/sdr_dout
        self.sdr_den_n     = Signal(byte_width)
        self.sdr_din       = Signal(data_width)
        self.sdr_dout      = Signal(data_width)

        # configuration parameters, these need to match the SDRAM IC datasheet
        self.sdr_init_done       = Signal()  # Indicate SDRAM init Done
        self.cfg_req_depth       = Signal(2) # max request accepted 
        self.cfg_sdr_en          = Signal()  # Enable SDRAM controller
        self.cfg_sdr_mode_reg    = Signal(13)
        self.cfg_sdr_tras_d      = Signal(4) # Active to precharge delay
        self.cfg_sdr_trp_d       = Signal(4) # Precharge to active delay
        self.cfg_sdr_trcd_d      = Signal(4) # Active to R/W delay
        self.cfg_sdr_cas         = Signal(3) # SDRAM CAS Latency
        self.cfg_sdr_trcar_d     = Signal(4) # Auto-refresh period
        self.cfg_sdr_twr_d       = Signal(4) # Write recovery delay
        self.cfg_sdr_rfsh        = Signal(self.refresh_timer_sz)
        self.cfg_sdr_rfmax       = Signal(self.refresh_row_count)

        # pins resource
        self.pins = pins

    @classmethod
    def add_verilog_source(cls, verilog_src_dir, platform):
        # add each of the verilog sources, needed for when doing platform.build
        for fname in [ './core/sdrc_bank_ctl.v', './core/sdrc_bank_fsm.v',
                        './core/sdrc_bs_convert.v', './core/sdrc_core.v',
                        './core/sdrc_req_gen.v', './core/sdrc_xfr_ctl.v',
                        './core/sdrc_define.v',
                        './lib/async_fifo.v', './lib/sync_fifo.v',
                        './top/sdrc_top.v', './wb2sdrc/wb2sdrc.v',
                     ]:
            # prepend the src directory to each filename, add its contents
            fullname = os.path.join(verilog_src_dir, fname)
            with open(fullname) as f:
                platform.add_file(fullname, f)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        # create definition of external verilog 16550 uart here, so that                # nmigen understands I/O directions (defined by i_ and o_ prefixes)
        bus = self.bus

        params = {
            # clock/reset (use DomainRenamer if needed)
            'i_wb_clk_i' : ClockSignal(),
            'i_wb_rst_i' : ResetSignal(),

            # wishbone bus signals
            'i_wb_adr_i' : bus.adr,
            'i_wb_dat_i' : bus.dat_w,
            'i_wb_sel_i' : bus.sel,
            'o_wb_dat_o' : bus.dat_r,
            'i_wb_we_i' : bus.we,
            'i_wb_stb_i' : bus.stb,
            'i_wb_cyc_i' : bus.cyc,
            'o_wb_ack_o' : bus.ack,

            # SDRAM signals
            'i_sdram_clk'      :  self.sdram_clk,
            'i_sdram_resetn'   :  self.sdram_resetn,
            'o_sdr_cs_n'       :  self.sdr_cs_n,
            'o_sdr_cke'        :  self.sdr_cke,
            'o_sdr_ras_n'      :  self.sdr_ras_n,
            'o_sdr_cas_n'      :  self.sdr_cas_n,
            'o_sdr_we_n'       :  self.sdr_we_n,
            'o_sdr_dqm'        :  self.sdr_dqm,
            'o_sdr_ba'         :  self.sdr_ba,
            'o_sdr_addr'       :  self.sdr_addr,
            'o_sdr_den_n'      : self.sdr_den_n,
            'i_sdr_din'        : self.sdr_din,
            'o_sdr_dout'       : self.sdr_dout,

            # configuration parameters (from the SDRAM IC datasheet)
            'o_sdr_init_done'      : self.sdr_init_done       ,
            'i_cfg_req_depth'      : self.cfg_req_depth       ,
            'i_cfg_sdr_en'         : self.cfg_sdr_en          ,
            'i_cfg_sdr_mode_reg'   : self.cfg_sdr_mode_reg    ,
            'i_cfg_sdr_tras_d'     : self.cfg_sdr_tras_d      ,
            'i_cfg_sdr_trp_d'      : self.cfg_sdr_trp_d       ,
            'i_cfg_sdr_trcd_d'     : self.cfg_sdr_trcd_d      ,
            'i_cfg_sdr_cas'        : self.cfg_sdr_cas         ,
            'i_cfg_sdr_trcar_d'    : self.cfg_sdr_trcar_d     ,
            'i_cfg_sdr_twr_d'      : self.cfg_sdr_twr_d       ,
            'i_cfg_sdr_rfsh'       : self.cfg_sdr_rfsh        ,
            'i_cfg_sdr_rfmax'      : self.cfg_sdr_rfmax,

            # verilog parameters
            'p_APP_AW'   : self.addr_width,    # Application Address Width
            'p_APP_DW'   : self.data_width,    # Application Data Width 
            'p_APP_BW'   : self.addr_width//8, # Application Byte Width
            'p_APP_RW'   : 9,                  # Application Request Width
            'p_SDR_DW'   : self.sdr_data_width,    # SDR Data Width 
            'p_SDR_BW'   : self.sdr_data_width//8, # SDR Byte Width
            'p_dw'       : self.data_width,    # data width
            'p_tw'       : 8,   # tag id width
            'p_bl'       : 9,   # burst_length_width 
        }
        m.submodules['sdrc_top'] = Instance("sdrc_top", **params)

        return m

        if self.pins is not None:
            comb += self.pins.tx.eq(self.tx_o)
            comb += self.rx_i.eq(self.pins.rx)

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
    sdram = SDRAM(name="sdram", data_width=8)
    create_ilang(sdram, [sdram.bus.cyc, sdram.bus.stb, sdram.bus.ack,
                         sdram.bus.dat_r, sdram.bus.dat_w, sdram.bus.adr,
                         sdram.bus.we, sdram.bus.sel,
                         sdram.sdram_clk, sdram.sdram_resetn,
                         sdram.sdr_cs_n, sdram.sdr_cke,
                         sdram.sdr_ras_n, sdram.sdr_cas_n, sdram.sdr_we_n,
                         sdram.sdr_dqm, sdram.sdr_ba, sdram.sdr_addr,
                         sdram.sdr_den_n, sdram.sdr_din, sdram.sdr_dout,
                         sdram.sdr_init_done, sdram.cfg_req_depth,
                         sdram.cfg_sdr_en, sdram.cfg_sdr_mode_reg,
                         sdram.cfg_sdr_tras_d, sdram.cfg_sdr_trp_d,
                         sdram.cfg_sdr_trcd_d, sdram.cfg_sdr_cas,
                         sdram.cfg_sdr_trcar_d, sdram.cfg_sdr_twr_d,
                         sdram.cfg_sdr_rfsh, sdram.cfg_sdr_rfmax,
                       ], "sdram")

