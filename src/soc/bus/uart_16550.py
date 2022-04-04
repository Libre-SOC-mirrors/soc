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
import tempfile

__all__ = ["UART16550"]


class UART16550(Elaboratable):
    """16550 UART from opencores, nmigen wrapper.  remember to call
       UART16550.add_verilog_source
    """

    def __init__(self, bus=None, features=None, name=None, data_width=32,
                       pins=None):
        if name is not None:
            # convention: give the name in the format "name_number"
            self.idx = int(name.split("_")[-1])
        else:
            self.idx = 0
            name = "uart_0"
        self.data_width = data_width

        # set up the wishbone bus
        if features is None:
            features = frozenset()
        if bus is None:
            bus = Interface(addr_width=5,
                            data_width=data_width,
                            features=features,
                            granularity=8,
                            name=name+"_wb_%d" % self.idx)
        self.bus = bus
        assert len(self.bus.dat_r) == data_width, \
                        "bus width must be %d" % data_width

        # IRQ for data buffer receive/xmit
        self.irq = Signal() 

        # 9-pin UART signals (if anyone still remembers those...)
        self.tx_o = Signal() # transmit
        self.rx_i = Signal() # receive
        self.rts_o = Signal() # ready to send
        self.cts_i = Signal() # clear to send
        self.dtr_o = Signal() # data terminal ready
        self.dsr_i = Signal() # data send ready
        self.ri_i = Signal() # can't even remember what this is!
        self.dcd_i = Signal() # or this!

        # pins resource
        self.pins = pins

    @classmethod
    def add_verilog_source(cls, verilog_src_dir, platform):
        # create a temp file containing "`define DATA_BUS_WIDTH_8"
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".v")
        t.write("`define DATA_BUS_WIDTH_8\n".encode())
        t.flush()
        t.seek(0)
        platform.add_file(t.name, t)

        # add each of the verilog sources, needed for when doing platform.build
        for fname in ['raminfr.v', 'uart_defines.v', 'uart_rfifo.v',
                      'uart_top.v', 'timescale.v', 'uart_receiver.v',
                      'uart_sync_flops.v', 'uart_transmitter.v',
                      'uart_debug_if.v', 'uart_regs.v',
                      'uart_tfifo.v', 'uart_wb.v'
                     ]:
            # prepend the src directory to each filename, add its contents
            fullname = os.path.join(verilog_src_dir, fname)
            with open(fullname) as f:
                platform.add_file(fullname, f)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        # create definition of external verilog 16550 uart here, so that                # nmigen understands I/O directions (defined by i_ and o_ prefixes)
        idx, bus = self.idx, self.bus
        uart = Instance("uart_top",
                            # clock/reset (use DomainRenamer if needed)
                            i_wb_clk_i=ClockSignal(),
                            i_wb_rst_i=ResetSignal(),
                            # wishbone bus signals
                            i_wb_adr_i=bus.adr,
                            i_wb_dat_i=bus.dat_w,
                            i_wb_sel_i=bus.sel,
                            o_wb_dat_o=bus.dat_r,
                            i_wb_we_i=bus.we,
                            i_wb_stb_i=bus.stb,
                            i_wb_cyc_i=bus.cyc,
                            o_wb_ack_o=bus.ack,
                            # interrupt line
                            o_int_o=self.irq,
                            # 9-pin RS232/UART signals
                            o_stx_pad_o=self.tx_o,
                            i_srx_pad_i=self.rx_i,
                            o_rts_pad_o=self.rts_o,
                            i_cts_pad_i=self.cts_i,
                            o_dtr_pad_o=self.dtr_o,
                            i_dsr_pad_i=self.dsr_i,
                            i_ri_pad_i=self.ri_i,
                            i_dcd_pad_i=self.dcd_i
                            );

        m.submodules['uart16550_%d' % self.idx] = uart

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
    uart = UART16550(name="uart_0", data_width=8)
    create_ilang(uart, [uart.bus.cyc, uart.bus.stb, uart.bus.ack,
                        uart.bus.dat_r, uart.bus.dat_w, uart.bus.adr,
                        uart.bus.we, uart.bus.sel,
                        uart.irq,
                        uart.tx_o, uart.rx_i, uart.rts_o, uart.cts_i,
                        uart.dtr_o, uart.dsr_i, uart.ri_i, uart.dcd_i
                       ], "uart_0")

