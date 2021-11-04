"""
Based on microwatt mmu.vhdl

* https://bugs.libre-soc.org/show_bug.cgi?id=491
* https://bugs.libre-soc.org/show_bug.cgi?id=450
"""

from nmigen import Elaboratable, Module, Signal, Shape, unsigned, Cat, Mux
from nmigen import Record, Memory
from nmigen import Const
from soc.fu.mmu.pipe_data import MMUInputData, MMUOutputData, MMUPipeSpec
from nmutil.singlepipe import ControlBase
from nmutil.util import rising_edge

from soc.experiment.mmu import MMU

from openpower.consts import MSR
from openpower.decoder.power_fields import DecodeFields
from openpower.decoder.power_fieldsn import SignalBitRange
from openpower.decoder.power_decoder2 import decode_spr_num
from openpower.decoder.power_enums import MicrOp

from soc.experiment.mem_types import LoadStore1ToMMUType
from soc.experiment.mem_types import MMUToLoadStore1Type

from soc.fu.ldst.loadstore import LoadStore1, TestSRAMLoadStore1
from nmutil.util import Display

class FSMMMUStage(ControlBase):
    """FSM MMU

    FSM-based MMU: must call set_ldst_interface and pass in an instance
    of a LoadStore1.  this to comply with the ConfigMemoryPortInterface API

    this Function Unit is extremely unusual in that it actually stores a
    "thing" rather than "processes inputs and produces outputs".  hence
    why it has to be a FSM.  linking up LD/ST however is going to have
    to be done back in Issuer (or Core).  sorted: call set_ldst_interface
    """
    def __init__(self, pspec):
        super().__init__()
        self.pspec = pspec

        # set up p/n data
        self.p.i_data = MMUInputData(pspec)
        self.n.o_data = MMUOutputData(pspec)

        self.mmu = MMU()

        # debugging output for gtkw
        self.debug0 = Signal(4)
        self.illegal = Signal()

        # for SPR field number access
        i = self.p.i_data
        self.fields = DecodeFields(SignalBitRange, [i.ctx.op.insn])
        self.fields.create_specs()

    def set_ldst_interface(self, ldst):
        """must be called back in Core, after FUs have been set up.
        one of those will be the MMU (us!) but the LoadStore1 instance
        must be set up in ConfigMemoryPortInterface. sigh.
        """
        # incoming PortInterface
        self.ldst = ldst
        self.dcache = self.ldst.dcache
        self.pi = self.ldst.pi

    def elaborate(self, platform):
        assert hasattr(self, "dcache"), "remember to call set_ldst_interface"
        m = super().elaborate(platform)
        comb, sync = m.d.comb, m.d.sync
        dcache = self.dcache

        # link mmu and dcache together
        m.submodules.mmu = mmu = self.mmu
        ldst = self.ldst # managed externally: do not add here
        m.d.comb += dcache.m_in.eq(mmu.d_out) # MMUToDCacheType
        m.d.comb += mmu.d_in.eq(dcache.m_out) # DCacheToMMUType

        l_in, l_out = mmu.l_in, mmu.l_out
        d_in, d_out = dcache.d_in, dcache.d_out
        wb_out, wb_in = dcache.wb_out, dcache.wb_in

        # link ldst and MMU together
        comb += l_in.eq(ldst.m_out)
        comb += ldst.m_in.eq(l_out)

        i_data, o_data = self.p.i_data, self.n.o_data
        a_i, b_i, o, spr1_o = i_data.ra, i_data.rb, o_data.o, o_data.spr1
        op = i_data.ctx.op
        msr_i = op.msr
        spr1_i = i_data.spr1

        # these are set / got here *ON BEHALF* of LoadStore1
        dsisr, dar = ldst.dsisr, ldst.dar

        # busy/done signals
        busy = Signal()
        done = Signal()
        m.d.comb += self.n.o_valid.eq(busy & done)
        m.d.comb += self.p.o_ready.eq(~busy)

        # take copy of X-Form SPR field
        x_fields = self.fields.FormXFX
        spr = Signal(len(x_fields.SPR))
        comb += spr.eq(decode_spr_num(x_fields.SPR))

        # based on MSR bits, set priv and virt mode.  TODO: 32-bit mode
        comb += d_in.priv_mode.eq(~msr_i[MSR.PR])
        comb += d_in.virt_mode.eq(msr_i[MSR.DR])
        #comb += d_in.mode_32bit.eq(msr_i[MSR.SF]) # ?? err

        # ok so we have to "pulse" the MMU (or dcache) rather than
        # hold the valid hi permanently.  guess what this does...
        valid = Signal()
        blip = Signal()
        m.d.comb += blip.eq(rising_edge(m, valid))

        with m.If(~busy):
            with m.If(self.p.i_valid):
                sync += busy.eq(1)
        with m.Else():

            # based on the Micro-Op, we work out which of MMU or DCache
            # should "action" the operation.  one of MMU or DCache gets
            # enabled ("valid") and we twiddle our thumbs until it
            # responds ("done").

            # WIP: properly implement MicrOp.OP_MTSPR and MicrOp.OP_MFSPR

            with m.Switch(op.insn_type):
                with m.Case(MicrOp.OP_MTSPR):
                    # despite redirection this FU **MUST** behave exactly
                    # like the SPR FU.  this **INCLUDES** updating the SPR
                    # regfile because the CSV file entry for OP_MTSPR
                    # categorically defines and requires the expectation
                    # that the CompUnit **WILL** write to the regfile.
                    comb += spr1_o.data.eq(a_i)
                    comb += spr1_o.ok.eq(1)
                    # subset SPR: first check a few bits
                    # XXX NOTE this must now cover **FOUR** values: this
                    # test might not be adequate.  DSISR, DAR, PGTBL and PID
                    # must ALL be covered here.
                    with m.If(~spr[9] & ~spr[5]):
                        comb += self.debug0.eq(3)
                        #if matched update local cached value
                        #commented out because there is a driver conflict
                        #with m.If(spr[0]):
                        #    sync += dsisr.eq(a_i[:32])
                        #with m.Else():
                        #    sync += dar.eq(a_i)
                        comb += done.eq(1)
                    # pass it over to the MMU instead
                    with m.Else():
                        # PGTBL and PID
                        comb += self.debug0.eq(4)
                        # blip the MMU and wait for it to complete
                        comb += valid.eq(1)   # start "pulse"
                        comb += l_in.valid.eq(blip)   # start
                        comb += l_in.mtspr.eq(1)      # mtspr mode
                        comb += l_in.sprn.eq(spr)  # which SPR
                        comb += l_in.rs.eq(a_i)    # incoming operand (RS)
                        comb += done.eq(1) # FIXME l_out.done

                with m.Case(MicrOp.OP_MFSPR):
                    comb += o.data.eq(spr1_i)
                    comb += o.ok.eq(1)
                    comb += done.eq(1)

                with m.Case(MicrOp.OP_TLBIE):
                    # pass TLBIE request to MMU (spec: v3.0B p1034)
                    # note that the spr is *not* an actual spr number, it's
                    # just that those bits happen to match with field bits
                    # RIC, PRS, R
                    comb += Display("TLBIE: %i %i",spr,l_out.done)
                    comb += valid.eq(1)   # start "pulse"
                    comb += l_in.valid.eq(blip)   # start
                    comb += l_in.tlbie.eq(1)   # mtspr mode
                    comb += l_in.sprn.eq(spr)  # use sprn to send insn bits
                    comb += l_in.addr.eq(b_i)  # incoming operand (RB)
                    comb += done.eq(l_out.done) # zzzz
                    comb += self.debug0.eq(2)

                with m.Case(MicrOp.OP_ILLEGAL):
                    comb += self.illegal.eq(1)

            with m.If(self.n.i_ready & self.n.o_valid):
                sync += busy.eq(0)

        return m

    def __iter__(self):
        yield from self.p
        yield from self.n

    def ports(self):
        return list(self)
