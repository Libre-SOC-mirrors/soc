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
    parser.add_argument("--pc-reset", default="0",
                        help="Set PC at reset (default 0)")
    parser.add_argument("--xlen", default=64, type=int,
                        help="Set register width [default 64]")
    # create a module that's directly compatible as a drop-in replacement
    # in microwatt.v
    parser.add_argument("--microwatt-compat", dest='mwcompat',
                        action="store_true",
                        help="generate microwatt-compatible interface",
                        default=False)
    parser.add_argument("--old-microwatt-compat", dest='old_mwcompat',
                        action="store_true",
                        help="generate old microwatt-compatible interface",
                        default=True)
    parser.add_argument("--microwatt-debug", dest='mwdebug',
                        action="store_true",
                        help="generate old microwatt-compatible interface",
                        default=False)
    # small cache option
    parser.add_argument("--small-cache", dest='smallcache',
                        action="store_true",
                        help="generate small caches",
                        default=False)

    # allow overlaps in TestIssuer
    parser.add_argument("--allow-overlap", dest='allow_overlap',
                        action="store_true",
                        help="allow overlap in TestIssuer",
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

    # default MSR
    msr_reset = (1<<MSR.LE) | (1<<MSR.SF) # 64-bit, little-endian default

    # default PC
    if args.pc_reset.startswith("0x"):
        pc_reset = int(args.pc_reset, 16)
    else:
        pc_reset = int(args.pc_reset)

    pspec = TestMemPspec(ldst_ifacetype=ldst_ifacetype,
                         imem_ifacetype=imem_ifacetype,
                         addr_wid=64,
                         mask_wid=8,
                         # pipeline and integer register file width
                         XLEN=args.xlen,
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
                         microwatt_old=args.old_mwcompat, # old microwatt api
                         microwatt_debug=args.mwdebug, # microwatt debug signals
                         small_cache=args.smallcache, # small cache/TLB sizes
                         allow_overlap=args.allow_overlap, # allow overlap
                         units=units,
                         msr_reset=msr_reset,
                         pc_reset=pc_reset)
    #if args.mwcompat:
    #    pspec.core_domain = 'sync'

    print("mmu", pspec.__dict__["microwatt_mmu"])
    print("nocore", pspec.__dict__["nocore"])
    print("regreduce", pspec.__dict__["regreduce"])
    print("gpio", pspec.__dict__["gpio"])
    print("sram4x4kblock", pspec.__dict__["sram4x4kblock"])
    print("xics", pspec.__dict__["xics"])
    print("use_pll", pspec.__dict__["use_pll"])
    print("debug", pspec.__dict__["debug"])
    print("SVP64", pspec.__dict__["svp64"])
    print("XLEN", pspec.__dict__["XLEN"])
    print("MSR@reset", hex(pspec.__dict__["msr_reset"]))
    print("PC@reset", hex(pspec.__dict__["pc_reset"]))
    print("Microwatt compatibility", pspec.__dict__["microwatt_compat"])
    print("Old Microwatt compatibility", pspec.__dict__["microwatt_old"])
    print("Microwatt debug", pspec.__dict__["microwatt_debug"])
    print("Small Cache/TLB", pspec.__dict__["small_cache"])

    if args.mwcompat:
        dut = TestIssuerInternal(pspec)
        name = "external_core_top"
    else:
        dut = TestIssuer(pspec)
        name = "test_issuer"

    vl = verilog.convert(dut, ports=dut.external_ports(), name=name)
    with open(args.output_filename, "w") as f:
        f.write(vl)
