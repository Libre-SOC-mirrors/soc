# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2021 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Funded by NLnet http://nlnet.nl
"""SVP64 RM (Remap) Record.

https://libre-soc.org/openpower/sv/svp64/

| Field Name  | Field bits | Description                            |
|-------------|------------|----------------------------------------|
| MASKMODE    | `0`        | Execution (predication) Mask Kind      |
| MASK        | `1:3`      | Execution Mask                         |
| ELWIDTH     | `4:5`      | Element Width                          |
| ELWIDTH_SRC | `6:7`      | Element Width for Source               |
| SUBVL       | `8:9`      | Sub-vector length                      |
| EXTRA       | `10:18`    | context-dependent extra                |
| MODE        | `19:23`    | changes Vector behaviour               |
"""

from nmigen import Elaboratable, Module, Signal, Const
from soc.decoder.power_enums import (SVP64RMMode, Function, SVPtype,
                                    SVP64PredMode, SVP64sat)
from soc.consts import EXTRA3, SVP64MODE
from soc.sv.svp64 import SVP64Rec
from nmutil.util import sel


"""RM Mode
there are three Mode variants, two for LD/ST and one for everything else
https://libre-soc.org/openpower/sv/svp64/
https://libre-soc.org/openpower/sv/ldst/

LD/ST immed:
00	str	sz dz	normal mode
01	inv	CR-bit	Rc=1: ffirst CR sel
01	inv	els RC1	Rc=0: ffirst z/nonz
10	N	dz els	sat mode: N=0/1 u/s
11	inv	CR-bit	Rc=1: pred-result CR sel
11	inv	els RC1	Rc=0: pred-result z/nonz

LD/ST indexed:
00	0	sz dz	normal mode
00	1	rsvd	reserved
01	inv	CR-bit	Rc=1: ffirst CR sel
01	inv	dz RC1	Rc=0: ffirst z/nonz
10	N	sz dz	sat mode: N=0/1 u/s
11	inv	CR-bit	Rc=1: pred-result CR sel
11	inv	dz RC1	Rc=0: pred-result z/nonz

Arithmetic:
00	0	sz dz	normal mode
00	1	dz CRM	reduce mode (mapreduce), SUBVL=1
00	1	SVM CRM	subvector reduce mode, SUBVL>1
01	inv	CR-bit	Rc=1: ffirst CR sel
01	inv	dz RC1	Rc=0: ffirst z/nonz
10	N	sz dz	sat mode: N=0/1 u/s
11	inv	CR-bit	Rc=1: pred-result CR sel
11	inv	dz RC1	Rc=0: pred-result z/nonz
"""

class SVP64RMModeDecode(Elaboratable):
    def __init__(self, name=None):
        self.rm_in = SVP64Rec(name=name)
        self.fn_in = Signal(Function) # LD/ST is different
        self.ptype_in = Signal(SVPtype)
        self.rc_in = Signal()
        self.ldst_idx = Signal()

        # main mode (normal, reduce, saturate, ffirst, pred-result)
        self.mode = Signal(SVP64RMMode)

        # predication
        self.predmode = Signal(SVP64PredMode)
        self.srcpred = Signal(3) # source predicate
        self.dstpred = Signal(3) # destination predicate
        self.pred_sz = Signal(1) # predicate source zeroing
        self.pred_dz = Signal(1) # predicate dest zeroing
        
        self.saturate = Signal(SVP64sat)
        self.RC1 = Signal()
        self.cr_sel = Signal(2)
        self.inv = Signal(1)
        self.map_evm = Signal(1)
        self.map_crm = Signal(1)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        mode = self.rm_in.mode

        # decode pieces of mode
        is_ldst = Signal()
        comb += is_ldst.eq(self.fn_in == Function.LDST)
        mode2 = sel(m, mode, SVP64MODE.MOD2)
        with m.Switch(mode2):
            with m.Case(0): # needs further decoding (LDST no mapreduce)
                with m.If(is_ldst):
                    comb += self.mode.eq(SVP64RMMode.NORMAL)
                with m.Elif(mode[SVP64MODE.REDUCE]):
                    comb += self.mode.eq(SVP64RMMode.MAPREDUCE)
                with m.Else():
                    comb += self.mode.eq(SVP64RMMode.NORMAL)
            with m.Case(1):
                comb += self.mode.eq(SVP64RMMode.FFIRST) # fail-first
            with m.Case(2):
                comb += self.mode.eq(SVP64RMMode.SATURATE) # saturate
            with m.Case(3):
                comb += self.mode.eq(SVP64RMMode.PREDRES) # predicate result

        # extract zeroing
        with m.Switch(mode2):
            with m.Case(0): # needs further decoding (LDST no mapreduce)
                with m.If(is_ldst):
                    comb += self.pred_sz.eq(mode[SVP64MODE.SZ])
                    comb += self.pred_dz.eq(mode[SVP64MODE.DZ])
                with m.Elif(mode[SVP64MODE.REDUCE]):
                    with m.If(self.rm_in.subvl == Const(0, 2)): # no SUBVL
                        comb += self.pred_dz.eq(mode[SVP64MODE.DZ])
                with m.Else():
                    comb += self.pred_sz.eq(mode[SVP64MODE.SZ])
                    comb += self.pred_dz.eq(mode[SVP64MODE.DZ])
            with m.Case(1, 3):
                with m.If(is_ldst):
                    with m.If(~self.ldst_idx):
                        comb += self.pred_dz.eq(mode[SVP64MODE.DZ])
                with m.Elif(self.rc_in):
                    comb += self.pred_dz.eq(mode[SVP64MODE.DZ])
            with m.Case(2):
                with m.If(is_ldst & ~self.ldst_idx):
                    comb += self.pred_dz.eq(mode[SVP64MODE.DZ])
                with m.Else():
                    comb += self.pred_sz.eq(mode[SVP64MODE.SZ])
                    comb += self.pred_dz.eq(mode[SVP64MODE.DZ])

        # extract src/dest predicate.  use EXTRA3.MASK because EXTRA2.MASK
        # is in exactly the same bits
        srcmask = sel(m, self.rm_in.extra, EXTRA3.MASK)
        dstmask = self.rm_in.mask
        with m.If(self.ptype_in == SVPtype.P2):
            comb += self.srcpred.eq(srcmask)
        with m.Else():
            comb += self.srcpred.eq(dstmask)
        comb += self.dstpred.eq(dstmask)

        # identify predicate mode
        with m.If(self.rm_in.mmode == 1):
            comb += self.predmode.eq(SVP64PredMode.CR) # CR Predicate
        with m.Elif((self.srcpred == 0) & (self.dstpred == 0)):
            comb += self.predmode.eq(SVP64PredMode.ALWAYS) # No predicate
        with m.Else():
            comb += self.predmode.eq(SVP64PredMode.INT) # non-zero src: INT

        # TODO: detect zeroing mode, saturation mode, a few more.

        return m

