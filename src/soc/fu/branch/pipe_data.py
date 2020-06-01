"""
    Optional Register allocation listed below.  mandatory input
    (CompBROpSubset, CIA) not included.

    * CR is Condition Register (not an SPR)
    * SPR1 and SPR2 are all from the SPR regfile.  2 ports are needed

    insn       CR  SPR2  SPR1
    ----       --  ----  ----
    op_b       xx  xx     xx
    op_ba      xx  xx     xx
    op_bl      xx  xx     xx
    op_bla     xx  xx     xx
    op_bc      CR, xx,    CTR
    op_bca     CR, xx,    CTR
    op_bcl     CR, xx,    CTR
    op_bcla    CR, xx,    CTR
    op_bclr    CR, LR,    CTR
    op_bclrl   CR, LR,    CTR
    op_bcctr   CR, xx,    CTR
    op_bcctrl  CR, xx,    CTR
    op_bctar   CR, TAR,   CTR
    op_bctarl  CR, TAR,   CTR
"""

from nmigen import Signal, Const, Cat
from ieee754.fpcommon.getop import FPPipeContext
from soc.decoder.power_decoder2 import Data
from soc.fu.pipe_data import IntegerData, CommonPipeSpec
from soc.fu.branch.br_input_record import CompBROpSubset # TODO: replace


class BranchInputData(IntegerData):
    regspec = [('SPR', 'spr1', '0:63'),
               ('SPR', 'spr2', '0:63'),
               ('CR', 'cr', '0:3'),
               ('PC', 'cia', '0:63')]
    def __init__(self, pspec):
        super().__init__(pspec)
        # Note: for OP_BCREG, SPR1 will either be CTR, LR, or TAR
        # this involves the *decode* unit selecting the register, based
        # on detecting the operand being bcctr, bclr or bctar

        self.spr1 = Signal(64, reset_less=True) # see table above, SPR1
        self.spr2 = Signal(64, reset_less=True) # see table above, SPR2
        self.cr = Signal(4, reset_less=True)   # Condition Register(s) CR0-7
        self.cia = Signal(64, reset_less=True)  # Current Instruction Address

        # convenience variables.  not all of these are used at once
        self.ctr = self.srr0 = self.hsrr0 = self.spr1
        self.lr = self.tar = self.srr1 = self.hsrr1 = self.spr2

    def __iter__(self):
        yield from super().__iter__()
        yield self.spr1
        yield self.spr2
        yield self.cr
        yield self.cia

    def eq(self, i):
        lst = super().eq(i)
        return lst + [self.spr1.eq(i.spr1), self.spr2.eq(i.spr2),
                      self.cr.eq(i.cr), self.cia.eq(i.cia)]


class BranchOutputData(IntegerData):
    regspec = [('SPR', 'spr1', '0:63'),
               ('SPR', 'spr2', '0:63'),
               ('PC', 'nia', '0:63')]
    def __init__(self, pspec):
        super().__init__(pspec)
        self.spr1 = Data(64, name="spr1")
        self.spr2 = Data(64, name="spr2")
        self.nia = Data(64, name="nia")

        # convenience variables.
        self.ctr = self.spr1
        self.lr = self.tar = self.spr2

    def __iter__(self):
        yield from super().__iter__()
        yield from self.spr1
        yield from self.spr2
        yield from self.nia

    def eq(self, i):
        lst = super().eq(i)
        return lst + [self.spr1.eq(i.spr1), self.spr2.eq(i.spr2),
                      self.nia.eq(i.nia)]


class BranchPipeSpec(CommonPipeSpec):
    regspec = (BranchInputData.regspec, BranchOutputData.regspec)
    opsubsetkls = CompBROpSubset
    def rdflags(self, e): # in order of regspec
        cr1_en = e.read_cr1.ok # CR A
        spr1_ok = e.read_spr1.ok # SPR1
        spr2_ok = e.read_spr2.ok # SPR2
        return Cat(spr1_ok, spr2_ok, cr1_en, 1) # CIA CR SPR1 SPR2
