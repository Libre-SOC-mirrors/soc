"""simple core input data

"""

from nmigen import Signal

from openpower.sv.svp64 import SVP64Rec

from openpower.decoder.decode2execute1 import Decode2ToExecute1Type
from openpower.decoder.decode2execute1 import IssuerDecode2ToOperand
from soc.config.state import CoreState


class FetchInput:
    """FetchInput: the input to the Fetch Unit

    * pc - the current Program Counter

    pretty much it for now!

    """
    def __init__(self):

        self.pc = Signal(64)

    def eq(self, i):
        return [self.pc.eq(i.pc),
               ]


class FetchOutput:
    """FetchOutput: the output from the fetch unit: one single instruction

    * state.  this contains PC, MSR, and SVSTATE. this is crucial information.
      (TODO: bigendian_i should really be read from the relevant MSR bit)

    * the raw instruction.  no decoding has been done - at all.

      (TODO: provide a *pair* of raw instructions so that packet
       inspection can be done, and SVP64 decoding and future 64-bit
       prefix analysis carried out.  however right now that is *not*
       the focus)
    """
    def __init__(self): #, svp64_en):
        #self.svp64_en = svp64_en

        # state and raw instruction (and SVP64 ReMap fields)
        self.state = CoreState("core_fetched")
        self.raw_insn_i = Signal(32) # one raw instruction
        self.bigendian_i = Signal() # bigendian - TODO, set by MSR.BE

    def eq(self, i):
        return [self.state.eq(i.state),
                self.raw_insn_i.eq(i.raw_insn_i),
                self.bigendian_i.eq(i.bigendian_i),
               ]


class CoreInput:
    """CoreInput: this is the input specification for Signals coming into core.

    * state.  this contains PC, MSR, and SVSTATE. this is crucial information.
      (TODO: bigendian_i should really be read from the relevant MSR bit)

    * the previously-decoded instruction goes into the Decode2Execute1Type
      data structure. no need for Core to re-decode that.  however note
      that *satellite* decoders *are* part of Core.

    * the raw instruction. this is used by satellite decoders internal to
      Core, to provide Function-Unit-specific information.  really, they
      should be part of the actual ALU itself (in order to reduce wires),
      but hey.

    * other stuff is related to SVP64.  the 24-bit SV REMAP field containing
      Vector context, etc.
    """
    def __init__(self, pspec, svp64_en, regreduce_en):
        self.pspec = pspec
        self.svp64_en = svp64_en
        self.e = Decode2ToExecute1Type("core", opkls=IssuerDecode2ToOperand,
                                regreduce_en=regreduce_en)

        # SVP64 RA_OR_ZERO needs to know if the relevant EXTRA2/3 field is zero
        self.sv_a_nz = Signal()

        # state and raw instruction (and SVP64 ReMap fields)
        self.state = CoreState("core")
        self.raw_insn_i = Signal(32) # raw instruction
        self.bigendian_i = Signal() # bigendian - TODO, set by MSR.BE
        if svp64_en:
            self.sv_rm = SVP64Rec(name="core_svp64_rm") # SVP64 RM field
            self.is_svp64_mode = Signal() # set if SVP64 mode is enabled
            self.use_svp64_ldst_dec = Signal() # use alternative LDST decoder
            self.sv_pred_sm = Signal() # TODO: SIMD width
            self.sv_pred_dm = Signal() # TODO: SIMD width

    def eq(self, i):
        res = [self.e.eq(i.e),
                self.sv_a_nz.eq(i.sv_a_nz),
                self.state.eq(i.state),
                self.raw_insn_i.eq(i.raw_insn_i),
                self.bigendian_i.eq(i.bigendian_i),
               ]
        if not self.svp64_en:
            return res
        res += [ self.sv_rm.eq(i.sv_rm),
                self.is_svp64_mode.eq(i.is_svp64_mode),
                self.use_svp64_ldst_dec.eq(i.use_svp64_ldst_dec),
                self.sv_pred_sm.eq(i.sv_pred_sm),
                self.sv_pred_dm.eq(i.sv_pred_dm),
                ]
        return res


class CoreOutput:
    def __init__(self):
        # start/stop and terminated signalling
        self.core_terminate_o = Signal()  # indicates stopped
        self.busy_o = Signal(name="corebusy_o")  # ALU is busy, no input
        self.any_busy_o = Signal(name="any_busy_o")  # at least one ALU busy
        self.exc_happened = Signal()             # exception happened

    def eq(self, i):
        return [self.core_terminate_o.eq(i.core_terminate_o),
                self.busy_o.eq(i.busy_o),
                self.any_busy_o.eq(i.any_busy_o),
                self.exc_happened.eq(i.exc_happened),
               ]


