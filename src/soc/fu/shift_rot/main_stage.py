# License: LGPLv3+
# Copyright (C) 2020 Michael Nolan <mtnolan2640@gmail.com>
# Copyright (C) 2020 Luke Kenneth Casson Leighton <lkcl@lkcl.net>

# This stage is intended to do most of the work of executing shift
# instructions, as well as carry and overflow generation. This module
# however should not gate the carry or overflow, that's up to the
# output stage
from nmigen import (Module, Signal, Cat, Repl, Mux, Const)
from nmutil.pipemodbase import PipeModBase
from soc.fu.pipe_data import get_pspec_draft_bitmanip
from soc.fu.shift_rot.pipe_data import (ShiftRotOutputData,
                                        ShiftRotInputData)
from nmutil.lut import BitwiseLut
from nmutil.grev import GRev
from openpower.decoder.power_enums import MicrOp
from soc.fu.shift_rot.rotator import Rotator

from openpower.decoder.power_fields import DecodeFields
from openpower.decoder.power_fieldsn import SignalBitRange


class ShiftRotMainStage(PipeModBase):
    def __init__(self, pspec):
        super().__init__(pspec, "main")
        self.draft_bitmanip = get_pspec_draft_bitmanip(pspec)
        self.fields = DecodeFields(SignalBitRange, [self.i.ctx.op.insn])
        self.fields.create_specs()

    def ispec(self):
        return ShiftRotInputData(self.pspec)

    def ospec(self):
        return ShiftRotOutputData(self.pspec)

    def elaborate(self, platform):
        XLEN = self.pspec.XLEN
        m = Module()
        comb = m.d.comb
        op = self.i.ctx.op
        o = self.o.o

        bitwise_lut = None
        grev = None
        if self.draft_bitmanip:
            bitwise_lut = BitwiseLut(input_count=3, width=XLEN)
            m.submodules.bitwise_lut = bitwise_lut
            comb += bitwise_lut.inputs[0].eq(self.i.rb)
            comb += bitwise_lut.inputs[1].eq(self.i.ra)
            comb += bitwise_lut.inputs[2].eq(self.i.rc)
            # 6 == log2(64) because we have 64-bit values
            grev = GRev(log2_width=(XLEN-1).bit_length())
            m.submodules.grev = grev
            with m.If(op.is_32bit):
                # 32-bit, so input is lower 32-bits zero-extended
                comb += grev.input.eq(self.i.ra[0:32])
                # 32-bit, so we only feed in log2(32) == 5 bits
                comb += grev.chunk_sizes.eq(self.i.rb[0:5])
            with m.Else():
                comb += grev.input.eq(self.i.ra)
                comb += grev.chunk_sizes.eq(self.i.rb)

        # NOTE: the sh field immediate is read in by PowerDecode2
        # (actually DecodeRB), whereupon by way of rb "immediate" mode
        # it ends up in self.i.rb.

        # obtain me and mb fields from instruction.
        m_fields = self.fields.instrs['M']
        md_fields = self.fields.instrs['MD']
        mb = Signal(m_fields['MB'][0:-1].shape())
        me = Signal(m_fields['ME'][0:-1].shape())
        mb_extra = Signal(1, reset_less=True)
        comb += mb.eq(m_fields['MB'][0:-1])
        comb += me.eq(m_fields['ME'][0:-1])
        comb += mb_extra.eq(md_fields['mb'][0:-1][0])

        # set up microwatt rotator module
        m.submodules.rotator = rotator = Rotator(XLEN)
        comb += [
            rotator.me.eq(me),
            rotator.mb.eq(mb),
            rotator.mb_extra.eq(mb_extra),
            rotator.rs.eq(self.i.rs),
            rotator.ra.eq(self.i.a),
            rotator.shift.eq(self.i.rb),  # can also be sh (in immediate mode)
            rotator.is_32bit.eq(op.is_32bit),
            rotator.arith.eq(op.is_signed),
        ]

        comb += o.ok.eq(1)  # defaults to enabled

        # instruction rotate type
        mode = Signal(4, reset_less=True)
        comb += Cat(rotator.right_shift,
                    rotator.clear_left,
                    rotator.clear_right,
                    rotator.sign_ext_rs).eq(mode)

        # outputs from the microwatt rotator module
        comb += [o.data.eq(rotator.result_o),
                 self.o.xer_ca.data.eq(Repl(rotator.carry_out_o, 2))]

        with m.Switch(op.insn_type):
            with m.Case(MicrOp.OP_SHL):
                comb += mode.eq(0b0000)  # L-shift
            with m.Case(MicrOp.OP_SHR):
                comb += mode.eq(0b0001)  # R-shift
            with m.Case(MicrOp.OP_RLC):
                comb += mode.eq(0b0110)  # clear LR
            with m.Case(MicrOp.OP_RLCL):
                comb += mode.eq(0b0010)  # clear L
            with m.Case(MicrOp.OP_RLCR):
                comb += mode.eq(0b0100)  # clear R
            with m.Case(MicrOp.OP_EXTSWSLI):
                comb += mode.eq(0b1000)  # L-ext
            if self.draft_bitmanip:
                with m.Case(MicrOp.OP_TERNLOG):
                    # TODO: this only works for ternlogi, change to get lut
                    # value from register when we implement other variants
                    comb += bitwise_lut.lut.eq(self.fields.FormTLI.TLI[:])
                    comb += o.data.eq(bitwise_lut.output)
                    comb += self.o.xer_ca.data.eq(0)
                with m.Case(MicrOp.OP_GREV):
                    comb += o.data.eq(grev.output)
                    comb += self.o.xer_ca.data.eq(0)
            with m.Default():
                comb += o.ok.eq(0)  # otherwise disable

        ###### sticky overflow and context, both pass-through #####

        comb += self.o.xer_so.data.eq(self.i.xer_so)
        comb += self.o.ctx.eq(self.i.ctx)

        return m
