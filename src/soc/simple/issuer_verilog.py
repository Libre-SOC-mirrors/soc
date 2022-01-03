"""simple core issuer verilog generator
"""

import argparse
from nmigen.cli import verilog

from openpower.consts import MSR
from soc.config.test.test_loadstore import TestMemPspec
from soc.simple.issuer import TestIssuer, TestIssuerInternal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple core issuer " \
                                     "verilog generator")
    parser.add_argument("output_filename")
    parser.add_argument("--enable-xics", dest='xics', action="store_true",
                        help="Enable interrupts",
                        default=True)
    parser.add_argument("--disable-xics", dest='xics', action="store_false",
                        help="Disable interrupts",
                        default=False)
    parser.add_argument("--enable-lessports", dest='lessports',
                        action="store_true",
                        help="Enable less regfile ports",
                        default=True)
    parser.add_argument("--disable-lessports", dest='lessports',
                        action="store_false",
                        help="enable more regfile ports",
                        default=False)
    parser.add_argument("--enable-core", dest='core', action="store_true",
                        help="Enable main core",
                        default=True)
    parser.add_argument("--disable-core", dest='core', action="store_false",
                        help="disable main core",
                        default=False)
    parser.add_argument("--enable-mmu", dest='mmu', action="store_true",
                        help="Enable mmu",
                        default=False)
    parser.add_argument("--disable-mmu", dest='mmu', action="store_false",
                        help="Disable mmu",
                        default=False)
    parser.add_argument("--enable-pll", dest='pll', action="store_true",
                        help="Enable pll",
                        default=False)
    parser.add_argument("--disable-pll", dest='pll', action="store_false",
                        help="Disable pll",
                        default=False)
    parser.add_argument("--enable-testgpio", action="store_true",
                        help="Disable gpio pins",
                        default=False)
    parser.add_argument("--enable-sram4x4kblock", action="store_true",
                        help="Disable sram 4x4k block",
                        default=False)
    parser.add_argument("--debug", default="jtag", help="Select debug " \
                        "interface [jtag | dmi] [default jtag]")
    parser.add_argument("--enable-svp64", dest='svp64', action="store_true",
                        help="Enable SVP64",
                        default=True)
    parser.add_argument("--disable-svp64", dest='svp64', action="store_false",
                        help="disable SVP64",
                        default=False)
    # create a module that's directly compatible as a drop-in replacement
    # in microwatt.v
    parser.add_argument("--microwatt-compat", dest='mwcompat',
                        action="store_true",
                        help="generate microwatt-compatible interface",
                        default=False)

    args = parser.parse_args()

    # convenience: set some defaults
    if args.mwcompat:
        args.pll = False
        args.debug = 'dmi'
        args.core = True
        args.xics = False
        args.gpio = False
        args.sram4x4kblock = False
        args.svp64 = False

    print(args)

    units = {'alu': 1,
             'cr': 1, 'branch': 1, 'trap': 1,
             'logical': 1,
             'spr': 1,
             'div': 1,
             'mul': 1,
             'shiftrot': 1
            }
    if args.mmu:
        units['mmu'] = 1 # enable MMU

    # decide which memory type to configure
    if args.mmu:
        ldst_ifacetype = 'mmu_cache_wb'
        imem_ifacetype = 'mmu_cache_wb'
    else:
        ldst_ifacetype = 'bare_wb'
        imem_ifacetype = 'bare_wb'

    # default MSR (TODO, provide option to set default PC as well)
    msr_reset = (1<<MSR.LE) | (1<<MSR.SF) # 64-bit, little-endian default

    pspec = TestMemPspec(ldst_ifacetype=ldst_ifacetype,
                         imem_ifacetype=imem_ifacetype,
                         addr_wid=48,
                         mask_wid=8,
                         # must leave at 64
                         reg_wid=64,
                         # set to 32 for instruction-memory width=32
                         imem_reg_wid=64,
                         # set to 32 to make data wishbone bus 32-bit
                         #wb_data_wid=32,
                         xics=args.xics, # XICS interrupt controller
                         nocore=not args.core, # test coriolis2 ioring
                         regreduce = args.lessports, # less regfile ports
                         use_pll=args.pll,  # bypass PLL
                         gpio=args.enable_testgpio, # for test purposes
                         sram4x4kblock=args.enable_sram4x4kblock, # add SRAMs
                         debug=args.debug,      # set to jtag or dmi
                         svp64=args.svp64,      # enable SVP64
                         microwatt_mmu=args.mmu,         # enable MMU
                         microwatt_compat=args.mwcompat, # microwatt compatible
                         units=units,
                         msr_reset=msr_reset)
    if args.mwcompat:
        pspec.core_domain = 'sync'

    print("mmu", pspec.__dict__["microwatt_mmu"])
    print("nocore", pspec.__dict__["nocore"])
    print("regreduce", pspec.__dict__["regreduce"])
    print("gpio", pspec.__dict__["gpio"])
    print("sram4x4kblock", pspec.__dict__["sram4x4kblock"])
    print("xics", pspec.__dict__["xics"])
    print("use_pll", pspec.__dict__["use_pll"])
    print("debug", pspec.__dict__["debug"])
    print("SVP64", pspec.__dict__["svp64"])
    print("Microwatt compatibility", pspec.__dict__["microwatt_compat"])

    if args.mwcompat:
        dut = TestIssuerInternal(pspec)
        name = "external_core_top"
    else:
        dut = TestIssuer(pspec)
        name = "test_issuer"

    vl = verilog.convert(dut, ports=dut.external_ports(), name=name)
    with open(args.output_filename, "w") as f:
        f.write(vl)
