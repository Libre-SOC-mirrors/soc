"""Decode2ToExecute1Type

based on Anton Blanchard microwatt decode2.vhdl

"""
from nmigen import Signal, Record
from nmutil.iocontrol import RecordObject
from soc.decoder.power_enums import (MicrOp, CryIn, Function,
                                     SPRfull, SPRreduced, LDSTMode)
from soc.consts import TT
from soc.experiment.mem_types import LDSTException


class Data(Record):

    def __init__(self, width, name):
        name_ok = "%s_ok" % name
        layout = ((name, width), (name_ok, 1))
        Record.__init__(self, layout)
        self.data = getattr(self, name) # convenience
        self.ok = getattr(self, name_ok) # convenience
        self.data.reset_less = True # grrr
        self.reset_less = True # grrr

    def ports(self):
        return [self.data, self.ok]


class IssuerDecode2ToOperand(RecordObject):
    """IssuerDecode2ToOperand

    contains the subset of fields needed for Issuer to decode the instruction
    and get register rdflags signals set up.  it also doubles up as the
    "Trap" temporary store, because part of the Decoder's job is to
    identify whether a trap / interrupt / exception should occur.
    """

    def __init__(self, name=None):

        RecordObject.__init__(self, name=name)

        # current "state" (TODO: this in its own Record)
        self.msr = Signal(64, reset_less=True)
        self.cia = Signal(64, reset_less=True)

        # instruction, type and decoded information
        self.insn = Signal(32, reset_less=True) # original instruction
        self.insn_type = Signal(MicrOp, reset_less=True)
        self.fn_unit = Signal(Function, reset_less=True)
        self.lk = Signal(reset_less=True)
        self.rc = Data(1, "rc")
        self.oe = Data(1, "oe")
        self.input_carry = Signal(CryIn, reset_less=True)
        self.traptype  = Signal(TT.size, reset_less=True) # trap main_stage.py
        self.ldst_exc  = LDSTException("exc")
        self.trapaddr  = Signal(13, reset_less=True)
        self.read_cr_whole = Data(8, "cr_rd") # CR full read mask
        self.write_cr_whole = Data(8, "cr_wr") # CR full write mask
        self.is_32bit = Signal(reset_less=True)


class Decode2ToOperand(IssuerDecode2ToOperand):

    def __init__(self, name=None):

        IssuerDecode2ToOperand.__init__(self, name=name)

        # instruction, type and decoded information
        self.imm_data = Data(64, name="imm")
        self.invert_in = Signal(reset_less=True)
        self.zero_a = Signal(reset_less=True)
        self.output_carry = Signal(reset_less=True)
        self.input_cr = Signal(reset_less=True)  # instr. has a CR as input
        self.output_cr = Signal(reset_less=True) # instr. has a CR as output
        self.invert_out = Signal(reset_less=True)
        self.is_32bit = Signal(reset_less=True)
        self.is_signed = Signal(reset_less=True)
        self.data_len = Signal(4, reset_less=True) # bytes
        self.byte_reverse  = Signal(reset_less=True)
        self.sign_extend  = Signal(reset_less=True)# do we need this?
        self.ldst_mode  = Signal(LDSTMode, reset_less=True) # LD/ST mode
        self.write_cr0 = Signal(reset_less=True)


class Decode2ToExecute1Type(RecordObject):

    def __init__(self, name=None, asmcode=True, opkls=None, do=None,
                       regreduce_en=False):

        if regreduce_en:
            SPR = SPRreduced
        else:
            SPR = SPRfull

        if do is None and opkls is None:
            opkls = Decode2ToOperand

        RecordObject.__init__(self, name=name)

        if asmcode:
            self.asmcode = Signal(8, reset_less=True) # only for simulator
        self.write_reg = Data(7, name="rego")
        self.write_ea = Data(7, name="ea") # for LD/ST in update mode
        self.read_reg1 = Data(7, name="reg1")
        self.read_reg2 = Data(7, name="reg2")
        self.read_reg3 = Data(7, name="reg3")
        self.write_spr = Data(SPR, name="spro")
        self.read_spr1 = Data(SPR, name="spr1")
        #self.read_spr2 = Data(SPR, name="spr2") # only one needed

        self.xer_in = Signal(3, reset_less=True)   # xer might be read
        self.xer_out = Signal(reset_less=True)  # xer might be written

        self.read_fast1 = Data(3, name="fast1")
        self.read_fast2 = Data(3, name="fast2")
        self.write_fast1 = Data(3, name="fasto1")
        self.write_fast2 = Data(3, name="fasto2")

        self.read_cr1 = Data(7, name="cr_in1")
        self.read_cr2 = Data(7, name="cr_in2")
        self.read_cr3 = Data(7, name="cr_in2")
        self.write_cr = Data(7, name="cr_out")

        # decode operand data
        print ("decode2execute init", name, opkls, do)
        #assert name is not None, str(opkls)
        if do is not None:
            self.do = do
        else:
            self.do = opkls(name)
