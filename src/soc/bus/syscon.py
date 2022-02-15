#!/usr/bin/env python3
#
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2022 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Sponsored by NLnet and NGI POINTER under EU Grants 871528 and 957073
# Part of the Libre-SOC Project.
#
# this is a System Console peripheral compatible with microwatt
# https://github.com/antonblanchard/microwatt/blob/master/syscon.vhdl

from nmigen import (Elaboratable, Cat, Module, Signal)
from nmigen.cli import rtlil, verilog

from lambdasoc.periph import Peripheral

__all__ = ["MicrowattSYSCON"]


class MicrowattSYSCON(Peripheral, Elaboratable):
    """Microwatt-compatible (Sys)tem (Con)figuration module
    """

    def __init__(self, *, sys_clk_freq=100e6,
                          has_uart=True,
                          uart_is_16550=True
                          ):
        super().__init__(name="syscon")
        self.sys_clk_freq = sys_clk_freq
        self.has_uart = has_uart
        self.uart_is_16550 = uart_is_16550

        # System control ports
        self.dram_at_0 = Signal()
        self.core_reset = Signal()
        self.soc_reset = Signal()

        # set up a CSR Bank and associated bridge. has to be in this order
        # (declare bank, declare bridge) for some unknown reason.
        # (r)ead regs will have a r_stb and r_data Record entry
        # (w)rite regs will have a w_stb and w_data Record entry
        bank = self.csr_bank()
        self._reg_sig_r       = bank.csr(64, "r") # signature
        self._reg_info_r      = bank.csr(64, "r") # info
        self._bram_info_r     = bank.csr(64, "r") # bram info
        self._dram_info_r     = bank.csr(64, "r") # dram info
        self._clk_info_r      = bank.csr(64, "r") # clock frequency
        self._ctrl_info_r     = bank.csr(64, "rw") # control info
        self._dram_init_r     = bank.csr(64, "r") # dram initialisation info
        self._spiflash_info_r = bank.csr(64, "r") # spi flash info
        self._uart0_info_r    = bank.csr(64, "r") # UART0 info (baud etc.)
        self._uart1_info_r    = bank.csr(64, "r") # UART1 info (baud etc.)
        self._bram_bootaddr_r = bank.csr(64, "r") # BRAM boot address

        # bridge the above-created CSRs over wishbone.  ordering and size
        # above mattered, the bridge automatically packs them together
        # as memory-addressable "things" for us
        self._bridge = self.bridge(data_width=32, granularity=8, alignment=3)
        self.bus = self._bridge.bus

    def elaborate(self, platform):
        m = Module()
        comb, sync = m.d.comb, m.d.comb
        m.submodules.bridge = self._bridge

        # enter data into the CSRs. r_data can be left live all the time,
        # w_data obviously has to be set only when w_stb triggers.

        # identifying signature
        comb += self._reg_sig_r.r_data.eq(0xf00daa5500010001)

        # system clock rate (hz)
        comb += self._clk_info_r.r_data.eq(int(self.sys_clk_freq)) # in hz

        # uart peripheral clock rate, currently assumed to be system clock
        # 0 ..31  : UART clock freq (in HZ)
        #     32  : UART is 16550 (otherwise pp)
        comb += self._uart0_info_r.r_data[0:32].eq(int(self.sys_clk_freq))
        comb += self._uart0_info_r.r_data[32].eq(1)

        # Reg Info, defines what peripherals and characteristics are present
        comb += self._reg_info_r.r_data[0].eq(self.has_uart) # has UART0
        comb += self._reg_info_r.r_data[5].eq(1)             # Large SYSCON

        # system control
        sysctrl = Cat(self.dram_at_0, self.core_reset, self.soc_reset)
        with m.If(self._ctrl_info_r.w_stb):
            sync += sysctrl.eq(self._ctrl_info_r.w_data)
        comb += self._ctrl_info_r.r_data.eq(sysctrl)

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
    from nmigen_soc import wishbone
    class QuickDemo(Elaboratable):
        def elaborate(self, platform):
            m = Module()
            arbiter = wishbone.Arbiter(addr_width=30, data_width=32,
                                       granularity=8)
            decoder = wishbone.Decoder(addr_width=30, data_width=32, 
                                       granularity=8)
            m.submodules.syscon = syscon = MicrowattSYSCON()
            m.submodules.decoder = decoder
            m.submodules.arbiter = arbiter
            decoder.add(syscon.bus, addr=0xc0000000)
            m.d.comb += arbiter.bus.connect(decoder.bus)
            return m
    m = QuickDemo()
    create_ilang(m, None, "syscondemo")

