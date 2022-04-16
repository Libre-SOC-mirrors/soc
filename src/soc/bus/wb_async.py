# Copyright (C) 2022 Raptor Engineering, LLC <support@raptorengineering.com>
#
# Based partly on code from LibreSoC
#
# Modifications for the Libre-SOC Project funded by NLnet and NGI POINTER
# under EU Grants 871528 and 957073, under the LGPLv3+ License
#
# this is a wrapper around the Verilog Wishbone Components wb_async_reg module

from nmigen import (Elaboratable, Cat, Module, Signal, ClockSignal, Instance,
                    ResetSignal, Const)

from nmigen_soc.wishbone.bus import Interface
from nmigen_soc.memory import MemoryMap
from nmigen.utils import log2_int
from nmigen.cli import rtlil, verilog
from nmutil.byterev import byte_reverse
import os

__all__ = ["WBAsyncBridge"]


class WBAsyncBridge(Elaboratable):
    """Verilog Wishbone Components wb_async_reg module, nmigen wrapper.
    remember to call WBAsyncBridge.add_verilog_source
    """

    def __init__(self, master_bus=None, slave_bus=None, master_features=None,
                       slave_features=None, name=None,
                       address_width=30, data_width=32, granularity=8,
                       master_clock_domain=None, slave_clock_domain=None):
        if name is not None:
            # convention: give the name in the format "name_number"
            self.idx = int(name.split("_")[-1])
        else:
            self.idx = 0
            name = "wbasyncbridge_0"
        self.address_width = address_width
        self.data_width = data_width
        self.granularity = granularity
        self.dsize = log2_int(self.data_width//self.granularity)

        # set up the clock domains
        if master_clock_domain is None:
            self.wb_mclk = ClockSignal()
            self.wb_mrst = ResetSignal()
        else:
            self.wb_mclk = ClockSignal(master_clock_domain)
            self.wb_mrst = ResetSignal(master_clock_domain)
        if slave_clock_domain is None:
            self.wb_sclk = ClockSignal()
            self.wb_srst = ResetSignal()
        else:
            self.wb_sclk = ClockSignal(slave_clock_domain)
            self.wb_srst = ResetSignal(slave_clock_domain)

        # set up the wishbone busses
        if master_features is None:
            master_features = frozenset()
        if slave_features is None:
            slave_features = frozenset()
        if master_bus is None:
            master_bus = Interface(addr_width=self.address_width,
                            data_width=self.data_width,
                            features=master_features,
                            granularity=self.granularity,
                            name=name+"_wb_%d_master" % self.idx)
        if slave_bus is None:
            slave_bus = Interface(addr_width=self.address_width,
                            data_width=self.data_width,
                            features=slave_features,
                            granularity=self.granularity,
                            name=name+"_wb_%d_slave" % self.idx)
        self.master_bus = master_bus
        assert len(self.master_bus.dat_r) == data_width, \
                        "bus width must be %d" % data_width
        self.slave_bus = slave_bus
        assert len(self.slave_bus.dat_r) == data_width, \
                        "bus width must be %d" % data_width

    @classmethod
    def add_verilog_source(cls, verilog_src_dir, platform):
        # add each of the verilog sources, needed for when doing platform.build
        for fname in ['wb_async_reg.v']:
            # prepend the src directory to each filename, add its contents
            fullname = os.path.join(verilog_src_dir, fname)
            with open(fullname) as f:
                platform.add_file(fullname, f)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        master_bus, slave_bus = self.master_bus, self.slave_bus
        slave_err = Signal()
        slave_rty = Signal()

        # create definition of external verilog bridge code here, so that
        # nmigen understands I/O directions (defined by i_ and o_ prefixes)
        idx = self.idx
        wb_async_bridge = Instance("wb_async_reg",
                            # Parameters
                            p_ADDR_WIDTH=self.address_width,
                            p_DATA_WIDTH=self.data_width,
                            # width of select is the data width
                            # *divided* by the data granularity.
                            # data_width=32-bit, data granularity=8-bit,
                            # select_width ==> 32/8 ==> 4
                            p_SELECT_WIDTH=self.data_width//self.granularity,

                            # Clocks/resets
                            i_wbm_clk=self.wb_mclk,
                            i_wbm_rst=self.wb_mrst,
                            i_wbs_clk=self.wb_sclk,
                            i_wbs_rst=self.wb_srst,

                            # Master Wishbone bus signals
                            i_wbm_adr_i=self.master_bus.adr,
                            i_wbm_dat_i=self.master_bus.dat_w,
                            o_wbm_dat_o=self.master_bus.dat_r,
                            i_wbm_we_i=self.master_bus.we,
                            i_wbm_sel_i=self.master_bus.sel,
                            i_wbm_stb_i=self.master_bus.stb,
                            i_wbm_cyc_i=self.master_bus.cyc,
                            o_wbm_ack_o=self.master_bus.ack,
                            #o_wbm_err=self.master_bus.err,
                            #o_wbm_rty_i=self.master_bus.rty,

                            # Slave Wishbone bus signals
                            o_wbs_adr_o=self.slave_bus.adr,
                            i_wbs_dat_i=self.slave_bus.dat_r,
                            o_wbs_dat_o=self.slave_bus.dat_w,
                            o_wbs_we_o=self.slave_bus.we,
                            o_wbs_sel_o=self.slave_bus.sel,
                            o_wbs_stb_o=self.slave_bus.stb,
                            o_wbs_cyc_o=self.slave_bus.cyc,
                            i_wbs_ack_i=self.slave_bus.ack,
                            i_wbs_err_i=slave_err,
                            i_wbs_rty_i=slave_rty
                            );

        # Wire unused signals to 0
        comb += slave_err.eq(0)
        comb += slave_rty.eq(0)

        m.submodules['wb_async_bridge_%d' % self.idx] = wb_async_bridge

        return m

    def ports(self):
        return [self.master_bus.adr, self.master_bus.dat_w,
                        self.master_bus.dat_r,
                        self.master_bus.we, self.master_bus.sel,
                        self.master_bus.stb,
                        self.master_bus.cyc, self.master_bus.ack,
                        self.master_bus.err,
                        self.master_bus.rty,
                        self.slave_bus.adr, self.slave_bus.dat_w,
                        self.slave_bus.dat_r,
                        self.slave_bus.we, self.slave_bus.sel,
                        self.slave_bus.stb,
                        self.slave_bus.cyc, self.slave_bus.ack,
                        self.slave_bus.err,
                        self.slave_bus.rty
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
    wbasyncbridge = WBAsyncBridge(name="wbasyncbridge_0", address_width=30, data_width=32, granularity=8)
    create_ilang(wbasyncbridge, wbasyncbridge.ports(), "wbasyncbridge_0")
