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

__all__ = ["UART16550"]


class UART16550(Elaboratable):
    """16550 UART from opencores, nmigen wrapper
    """

    def __init__(self, bus=None, features=None, name=None):
        if name is not None:
            # convention: give the name in the format "name_number"
            self.idx = int(name.split("_")[-1])
        else:
            self.idx = 0

        # set up the wishbone bus
        if features is None:
            features = frozenset()
        if bus is None:
            bus = Interface(addr_width=5,
                            data_width=32,
                            features=features,
                            name=name+"_wb_%d" % self.idx)
        self.bus = bus
        assert len(self.bus.dat_r) == 32, "bus width must be 32"

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

    def elaborate(self, platform):
        m = Module()

        # create external verilog 16550 uart here
        idx, bus = self.idx, self.bus
        uart = Instance("uart_top_%d" % idx, 
                            i_wb_clk_i=ClockSignal(),
                            i_wb_rst_i=ResetSignal(),
                            i_wb_adr_i=bus.adr,
                            i_wb_dat_i=bus.dat_w,
                            i_wb_sel_i=bus.sel,
                            o_wb_dat_o=bus.dat_r,
                            i_wb_we_i=bus.we,
                            i_wb_stb_i=bus.stb,
                            i_wb_cyc_i=bus.cyc,
                            o_wb_ack_o=bus.ack,
                            o_int_o=self.irq,
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
    uart = UART16550(name="uart_0")
    create_ilang(uart, [uart.bus.cyc, uart.bus.stb, uart.bus.ack,
                        uart.bus.dat_r, uart.bus.dat_w, uart.bus.adr,
                        uart.bus.we, uart.bus.sel,
                        uart.irq,
                        uart.tx_o, uart.rx_i, uart.rts_o, uart.cts_i,
                        uart.dtr_o, uart.dsr_i, uart.ri_i, uart.dcd_i
                       ], "uart_0")

