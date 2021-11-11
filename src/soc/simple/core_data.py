"""simple core input data

"""

from nmigen import Signal

from openpower.sv.svp64 import SVP64Rec

from openpower.decoder.decode2execute1 import Decode2ToExecute1Type
from openpower.decoder.decode2execute1 import IssuerDecode2ToOperand
from soc.config.state import CoreState


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
        self.e.eq(i.e)
        self.sv_a_nz.eq(i.sv_a_nz)
        self.state.eq(i.state)
        self.raw_insn_i.eq(i.raw_insn_i)
        self.bigendian_i.eq(i.bigendian_i)
        if not self.svp64_en:
            return
        self.sv_rm.eq(i.sv_rm)
        self.is_svp64_mode.eq(i.is_svp64_mode)
        self.use_svp64_ldst_dec.eq(i.use_svp64_ldst_dec)
        self.sv_pred_sm.eq(i.sv_pred_sm)
        self.sv_pred_dm.eq(i.sv_pred_dm)


class CoreOutput:
    def __init__(self):
        # start/stop and terminated signalling
        self.core_terminate_o = Signal()  # indicates stopped
        self.busy_o = Signal(name="corebusy_o")  # at least one ALU busy
        self.exc_happened = Signal()             # exception happened

    def eq(self, i):
        self.core_terminate_o.eq(i.core_terminate_o)
        self.busy_o.eq(i.busy_o)
        self.exc_happened.eq(i.exc_happened)


