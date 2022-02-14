#!/usr/bin/env python3
#
# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2022 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Sponsored by NLnet and NGI POINTER under EU Grants 871528 and 957073
# Part of the Libre-SOC Project.
#
# this is a wrapper around the opencores verilog core16550 module

from nmigen import (Elaboratable, Cat, Module, Signal, ClockSignal, Instance,
                    ResetSignal, Const)
from nmigen.cli import rtlil, verilog

from soc.debug.dmi import DMIInterface
from nmigen_soc.wishbone.bus import Interface
import os

__all__ = ["ExternalCore"]


class ExternalCore(Elaboratable):
    """External Core verilog wrapper for microwatt and libre-soc
   (actually, anything prepared to map to the Signals defined below)
   remember to call ExternalCore.add_verilog_source
    """

    def __init__(self, ibus=None, dbus=None, features=None, name=None):

        # set up the icache wishbone bus
        if features is None:
            features = frozenset(("stall",))
        if ibus is None:
            ibus = Interface(addr_width=32,
                            data_width=64,
                            features=features,
                            granularity=8,
                            name="core_ibus")
        if dbus is None:
            dbus = Interface(addr_width=32,
                            data_width=64,
                            features=features,
                            granularity=8,
                            name="core_dbus")
        self.dmi = DMIInterface(name="dmi")
        self.ibus = ibus
        self.dbus = dbus

        assert len(self.ibus.dat_r) == 64, "bus width must be 64"
        assert len(self.dbus.dat_r) == 64, "bus width must be 64"

        # IRQ for data buffer receive/xmit
        self.irq = Signal() 

        # debug monitoring signals
        self.nia = Signal(64)
        self.nia_req = Signal()
        self.msr = Signal(64)
        self.ldst_addr = Signal(64)
        self.ldst_req = Signal()

        # alternative reset and termination indicator
        self.alt_reset = Signal()
        self.terminated_o = Signal()

    @classmethod
    def add_verilog_source(cls, verilog_src_dir, platform):
        # add each of the verilog sources, needed for when doing platform.build
        for fname in ['external_core_top.v',
                     ]:
            # prepend the src directory to each filename, add its contents
            fullname = os.path.join(verilog_src_dir, fname)
            with open(fullname) as f:
                platform.add_file(fullname, f)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        # create definition of external core here, so that
        # nmigen understands I/O directions (defined by i_ and o_ prefixes)
        ibus, dbus, dmi = self.ibus, self.dbus, self.dmi

        # sigh, microwatt wishbone address is borked, it contains the 3 LSBs
        ibus_adr = Signal(32)
        dbus_adr = Signal(32)
        m.d.comb += ibus.adr.eq(ibus_adr[3:])
        m.d.comb += dbus.adr.eq(dbus_adr[3:])

        kwargs = {
            # clock/reset signals
            'i_clk': ClockSignal(),
            'i_rst': ResetSignal(),
            # DMI interface
            'i_dmi_addr': dmi.addr_i,
            'i_dmi_req': dmi.req_i,
            'i_dmi_wr': dmi.we_i,
            'i_dmi_din': dmi.din,
            'o_dmi_dout': dmi.dout,
            'o_dmi_ack': dmi.ack_o,
            # debug/monitor signals
            'o_nia': self.nia,
            'o_nia_req': self.nia_req,
            'o_msr_o': self.msr,
            'o_ldst_addr': self.ldst_addr,
            'o_ldst_req': self.ldst_req,
            'i_alt_reset': self.alt_reset,
            'o_terminated_out': self.terminated_o,
            # wishbone instruction bus
            'o_wishbone_insn_out.adr': ibus_adr,
            'o_wishbone_insn_out.dat': ibus.dat_w,
            'o_wishbone_insn_out.sel': ibus.sel,
            'o_wishbone_insn_out.cyc': ibus.cyc,
            'o_wishbone_insn_out.stb': ibus.stb,
            'o_wishbone_insn_out.we': ibus.we,
            'i_wishbone_insn_in.dat': ibus.dat_r,
            'i_wishbone_insn_in.ack': ibus.ack,
            'i_wishbone_insn_in.stall': ibus.stall,
            # wishbone data bus
            'o_wishbone_data_out.adr': dbus_adr,
            'o_wishbone_data_out.dat': dbus.dat_w,
            'o_wishbone_data_out.sel': dbus.sel,
            'o_wishbone_data_out.cyc': dbus.cyc,
            'o_wishbone_data_out.stb': dbus.stb,
            'o_wishbone_data_out.we': dbus.we,
            'i_wishbone_data_in.dat': dbus.dat_r,
            'i_wishbone_data_in.ack': dbus.ack,
            'i_wishbone_data_in.stall': dbus.stall,
            # external interrupt request
            'i_ext_irq': self.irq,
        }
        core = Instance("external_core_top", **kwargs)
        m.submodules['core_top'] = core

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
    core = ExternalCore(name="core")
    create_ilang(core, [
                        core.ibus.cyc, core.ibus.stb, core.ibus.ack,
                        core.ibus.dat_r, core.ibus.dat_w, core.ibus.adr,
                        core.ibus.we, core.ibus.sel, core.ibus.stall,
                        core.dbus.cyc, core.dbus.stb, core.dbus.ack,
                        core.dbus.dat_r, core.dbus.dat_w, core.dbus.adr,
                        core.dbus.we, core.dbus.sel,
                        core.irq, core.alt_reset, core.terminated_o,
                        core.msr, core.nia, core.nia_req,
                        core.ldst_addr, core.ldst_req,
                        core.dmi.addr_i, core.dmi.req_i, core.dmi.we_i,
                        core.dmi.din, core.dmi.dout, core.dmi.ack_o,
                       ], "core_0")

