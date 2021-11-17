# License: LGPLv3+
# Copyright (C) 2020 Michael Nolan <mtnolan2640@gmail.com>
# Copyright (C) 2020 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Copyright (C) 2021 Jacob Lifshay <programmerjake@gmail.com>

# This stage is intended to do most of the work of executing bitmanip
# instructions, as well as overflow generation. This module however should not
# gate the overflow, that's up to the output stage
from nmigen.hdl.dsl import Module
from nmutil.pipemodbase import PipeModBase
from soc.fu.bitmanip.pipe_data import (BitManipOutputData,
                                       BitManipInputData)
from openpower.decoder.power_enums import MicrOp
from openpower.decoder.power_fields import DecodeFields
from openpower.decoder.power_fieldsn import SignalBitRange
from nmutil.lut import BitwiseLut


class BitManipMainStage(PipeModBase):
    def __init__(self, pspec):
        super().__init__(pspec, "main")
        self.fields = DecodeFields(SignalBitRange, [self.i.ctx.op.insn])
        self.fields.create_specs()

    def ispec(self):
        return BitManipInputData(self.pspec)

    def ospec(self):
        return BitManipOutputData(self.pspec)

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb
        op = self.i.ctx.op
        o = self.o.o

        bitwise_lut = BitwiseLut(input_count=3, width=64)
        m.submodules.bitwise_lut = bitwise_lut
        comb += bitwise_lut.inputs[0].eq(self.i.rb)
        comb += bitwise_lut.inputs[1].eq(self.i.ra)
        comb += bitwise_lut.inputs[2].eq(self.i.rc)

        comb += o.ok.eq(1)  # defaults to enabled

        with m.Switch(op.insn_type):
            with m.Case(MicrOp.OP_TERNLOG):
                # TODO: this only works for ternaryi, change to get lut value
                # from register when we implement other variants
                comb += bitwise_lut.lut.eq(self.fields.FormTLI.TLI)
                comb += o.data.eq(bitwise_lut.output)
            with m.Default():
                comb += o.ok.eq(0)  # otherwise disable

        ###### sticky overflow and context, both pass-through #####

        comb += self.o.xer_so.data.eq(self.i.xer_so)
        comb += self.o.ctx.eq(self.i.ctx)

        return m
