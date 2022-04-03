"""simple core

not in any way intended for production use.  connects up FunctionUnits to
Register Files in a brain-dead fashion that only permits one and only one
Function Unit to be operational.

the principle here is to take the Function Units, analyse their regspecs,
and turn their requirements for access to register file read/write ports
into groupings by Register File and Register File Port name.

under each grouping - by regfile/port - a list of Function Units that
need to connect to that port is created.  as these are a contended
resource a "Broadcast Bus" per read/write port is then also created,
with access to it managed by a PriorityPicker.

the brain-dead part of this module is that even though there is no
conflict of access, regfile read/write hazards are *not* analysed,
and consequently it is safer to wait for the Function Unit to complete
before allowing a new instruction to proceed.
(update: actually this is being added now:
https://bugs.libre-soc.org/show_bug.cgi?id=737)
"""

from nmigen import (Elaboratable, Module, Signal, ResetSignal, Cat, Mux,
                    Const)
from nmigen.cli import rtlil

from openpower.decoder.power_decoder2 import PowerDecodeSubset
from openpower.decoder.power_regspec_map import regspec_decode
from openpower.sv.svp64 import SVP64Rec

from nmutil.picker import PriorityPicker
from nmutil.util import treereduce
from nmutil.singlepipe import ControlBase

from soc.fu.compunits.compunits import AllFunctionUnits, LDSTFunctionUnit
from soc.regfile.regfiles import RegFiles
from openpower.decoder.power_decoder2 import get_rdflags
from soc.experiment.l0_cache import TstL0CacheBuffer  # test only
from soc.config.test.test_loadstore import TestMemPspec
from openpower.decoder.power_enums import MicrOp, Function
from soc.simple.core_data import CoreInput, CoreOutput

from collections import defaultdict, namedtuple
import operator

from nmutil.util import rising_edge

FUSpec = namedtuple("FUSpec", ["funame", "fu", "idx"])
ByRegSpec = namedtuple("ByRegSpec", ["okflag", "regport", "wid", "specs"])

# helper function for reducing a list of signals down to a parallel
# ORed single signal.
def ortreereduce(tree, attr="o_data"):
    return treereduce(tree, operator.or_, lambda x: getattr(x, attr))


def ortreereduce_sig(tree):
    return treereduce(tree, operator.or_, lambda x: x)


# helper function to place full regs declarations first
def sort_fuspecs(fuspecs):
    res = []
    for (regname, fspec) in fuspecs.items():
        if regname.startswith("full"):
            res.append((regname, fspec))
    for (regname, fspec) in fuspecs.items():
        if not regname.startswith("full"):
            res.append((regname, fspec))
    return res  # enumerate(res)


# a hazard bitvector "remap" function which returns an AST expression
# that remaps read/write hazard regfile port numbers to either a full
# bitvector or a reduced subset one.  SPR for example is reduced to a
# single bit.
# CRITICALLY-IMPORTANT NOTE: these bitvectors *have* to match up per
# regfile!  therefore the remapping is per regfile, *NOT* per regfile
# port and certainly not based on whether it is a read port or write port.
# note that any reductions here will result in degraded performance due
# to conflicts, but at least it keeps the hazard matrix sizes down to "sane"
def bitvector_remap(regfile, rfile, port):
    # 8-bits (at the moment, no SVP64), CR is unary: no remap
    if regfile == 'CR':
        return port
    # 3 bits, unary alrady: return the port
    if regfile == 'XER':
        return port
    # 3 bits, unary: return the port
    if regfile == 'XER':
        return port
    # 5 bits, unary: return the port
    if regfile == 'STATE':
        return port
    # 9 bits (9 entries), might be unary already
    if regfile == 'FAST':
        if rfile.unary: # FAST might be unary already
            return port
        else:
            return 1 << port
    # 10 bits (!!) - reduce to one
    if regfile == 'SPR':
        if rfile.unary: # FAST might be unary already
            return port
        else:
            return 1 << port
    if regfile == 'INT':
        if rfile.unary: # INT, check if unary/binary
            return port
        else:
            return 1 << port


# derive from ControlBase rather than have a separate Stage instance,
# this is simpler to do
class NonProductionCore(ControlBase):
    def __init__(self, pspec):
        self.pspec = pspec

        # test is SVP64 is to be enabled
        self.svp64_en = hasattr(pspec, "svp64") and (pspec.svp64 == True)

        # test to see if regfile ports should be reduced
        self.regreduce_en = (hasattr(pspec, "regreduce") and
                             (pspec.regreduce == True))

        # test to see if overlapping of instructions is allowed
        # (not normally enabled for TestIssuer FSM but useful for checking
        # the bitvector hazard detection, before doing In-Order)
        self.allow_overlap = (hasattr(pspec, "allow_overlap") and
                             (pspec.allow_overlap == True))

        # test core type
        self.make_hazard_vecs = self.allow_overlap
        self.core_type = "fsm"
        if hasattr(pspec, "core_type"):
            self.core_type = pspec.core_type

        super().__init__(stage=self)

        # single LD/ST funnel for memory access
        self.l0 = l0 = TstL0CacheBuffer(pspec, n_units=1)
        pi = l0.l0.dports[0]

        # function units (only one each)
        # only include mmu if enabled in pspec
        self.fus = AllFunctionUnits(pspec, pilist=[pi])

        # link LoadStore1 into MMU
        mmu = self.fus.get_fu('mmu0')
        ldst0 = self.fus.get_fu('ldst0')
        print ("core pspec", pspec.ldst_ifacetype)
        print ("core mmu", mmu)
        if mmu is not None:
            lsi = l0.cmpi.lsmem.lsi # a LoadStore1 Interface object
            print ("core lsmem.lsi", lsi)
            mmu.alu.set_ldst_interface(lsi)
            # urr store I-Cache in core so it is easier to get at
            self.icache = lsi.icache

        # alternative reset values for STATE regs
        self.msr_at_reset = 0x0
        self.pc_at_reset = 0x0
        if hasattr(pspec, "msr_reset") and isinstance(pspec.msr_reset, int):
            self.msr_at_reset = pspec.msr_reset
        if hasattr(pspec, "pc_reset") and isinstance(pspec.pc_reset, int):
            self.pc_at_reset = pspec.pc_reset
        state_resets = [self.pc_at_reset,  # PC at reset
                        self.msr_at_reset, # MSR at reset
                        0x0,               # SVSTATE at reset
                        0x0,               # DEC at reset
                        0x0]               # TB at reset

        # register files (yes plural)
        self.regs = RegFiles(pspec, make_hazard_vecs=self.make_hazard_vecs,
                                    state_resets=state_resets)

        # set up input and output: unusual requirement to set data directly
        # (due to the way that the core is set up in a different domain,
        # see TestIssuer.setup_peripherals
        self.p.i_data, self.n.o_data = self.new_specs(None)
        self.i, self.o = self.p.i_data, self.n.o_data

        # actual internal input data used (captured)
        self.ireg = self.ispec()

        # create per-FU instruction decoders (subsetted).  these "satellite"
        # decoders reduce wire fan-out from the one (main) PowerDecoder2
        # (used directly by the trap unit) to the *twelve* (or more)
        # Function Units.  we can either have 32 wires (the instruction)
        # to each, or we can have well over a 200 wire fan-out (to 12
        # ALUs). it's an easy choice to make.
        self.decoders = {}
        self.des = {}

        # eep, these should be *per FU* i.e. for FunctionUnitBaseMulti
        # they should be shared (put into the ALU *once*).

        for funame, fu in self.fus.fus.items():
            f_name = fu.fnunit.name
            fnunit = fu.fnunit.value
            opkls = fu.opsubsetkls
            if f_name == 'TRAP':
                # TRAP decoder is the *main* decoder
                self.trapunit = funame
                continue
            assert funame not in self.decoders
            self.decoders[funame] = PowerDecodeSubset(None, opkls, f_name,
                                                      final=True,
                                                      state=self.ireg.state,
                                            svp64_en=self.svp64_en,
                                            regreduce_en=self.regreduce_en)
            self.des[funame] = self.decoders[funame].do
            print ("create decoder subset", funame, opkls, self.des[funame])

        # create per-Function Unit write-after-write hazard signals
        # yes, really, this should have been added in ReservationStations
        # but hey.
        for funame, fu in self.fus.fus.items():
            fu._waw_hazard = Signal(name="waw_%s" % funame)

        # share the SPR decoder with the MMU if it exists
        if "mmu0" in self.decoders:
            self.decoders["mmu0"].mmu0_spr_dec = self.decoders["spr0"]

        # allow pausing of the DEC/TB FSM back in Issuer, by spotting
        # if there is an MTSPR instruction
        self.pause_dec_tb = Signal()

    # next 3 functions are Stage API Compliance
    def setup(self, m, i):
        pass

    def ispec(self):
        return CoreInput(self.pspec, self.svp64_en, self.regreduce_en)

    def ospec(self):
        return CoreOutput()

    # elaborate function to create HDL
    def elaborate(self, platform):
        m = super().elaborate(platform)

        # for testing purposes, to cut down on build time in coriolis2
        if hasattr(self.pspec, "nocore") and self.pspec.nocore == True:
            x = Signal() # dummy signal
            m.d.sync += x.eq(~x)
            return m
        comb = m.d.comb

        m.submodules.fus = self.fus
        m.submodules.l0 = l0 = self.l0
        self.regs.elaborate_into(m, platform)
        regs = self.regs
        fus = self.fus.fus

        # amalgamate write-hazards into a single top-level Signal
        self.waw_hazard = Signal()
        whaz = []
        for funame, fu in self.fus.fus.items():
            whaz.append(fu._waw_hazard)
        comb += self.waw_hazard.eq(Cat(*whaz).bool())

        # connect decoders
        self.connect_satellite_decoders(m)

        # ssh, cheat: trap uses the main decoder because of the rewriting
        self.des[self.trapunit] = self.ireg.e.do

        # connect up Function Units, then read/write ports, and hazard conflict
        self.issue_conflict = Signal()
        fu_bitdict, fu_selected = self.connect_instruction(m)
        raw_hazard = self.connect_rdports(m, fu_bitdict, fu_selected)
        self.connect_wrports(m, fu_bitdict, fu_selected)
        if self.allow_overlap:
            comb += self.issue_conflict.eq(raw_hazard)

        # note if an exception happened.  in a pipelined or OoO design
        # this needs to be accompanied by "shadowing" (or stalling)
        el = []
        for exc in self.fus.excs.values():
            el.append(exc.happened)
        if len(el) > 0: # at least one exception
            comb += self.o.exc_happened.eq(Cat(*el).bool())

        return m

    def connect_satellite_decoders(self, m):
        comb = m.d.comb
        for k, v in self.decoders.items():
            # connect each satellite decoder and give it the instruction.
            # as subset decoders this massively reduces wire fanout given
            # the large number of ALUs
            m.submodules["dec_%s" % k] = v
            comb += v.dec.raw_opcode_in.eq(self.ireg.raw_insn_i)
            comb += v.dec.bigendian.eq(self.ireg.bigendian_i)
            # sigh due to SVP64 RA_OR_ZERO detection connect these too
            comb += v.sv_a_nz.eq(self.ireg.sv_a_nz)
            if not self.svp64_en:
                continue
            comb += v.pred_sm.eq(self.ireg.sv_pred_sm)
            comb += v.pred_dm.eq(self.ireg.sv_pred_dm)
            if k == self.trapunit:
                continue
            comb += v.sv_rm.eq(self.ireg.sv_rm) # pass through SVP64 RM
            comb += v.is_svp64_mode.eq(self.ireg.is_svp64_mode)
            # only the LDST PowerDecodeSubset *actually* needs to
            # know to use the alternative decoder.  this is all
            # a terrible hack
            if not k.lower().startswith("ldst"):
                continue
            comb += v.use_svp64_ldst_dec.eq( self.ireg.use_svp64_ldst_dec)

    def connect_instruction(self, m):
        """connect_instruction

        uses decoded (from PowerOp) function unit information from CSV files
        to ascertain which Function Unit should deal with the current
        instruction.

        some (such as OP_ATTN, OP_NOP) are dealt with here, including
        ignoring it and halting the processor.  OP_NOP is a bit annoying
        because the issuer expects busy flag still to be raised then lowered.
        (this requires a fake counter to be set).
        """
        comb, sync = m.d.comb, m.d.sync
        fus = self.fus.fus

        # indicate if core is busy
        busy_o = self.o.busy_o
        any_busy_o = self.o.any_busy_o

        # connect up temporary copy of incoming instruction. the FSM will
        # either blat the incoming instruction (if valid) into self.ireg
        # or if the instruction could not be delivered, keep dropping the
        # latched copy into ireg
        ilatch = self.ispec()
        self.instr_active = Signal()

        # enable/busy-signals for each FU, get one bit for each FU (by name)
        fu_enable = Signal(len(fus), reset_less=True)
        fu_busy = Signal(len(fus), reset_less=True)
        fu_bitdict = {}
        fu_selected = {}
        for i, funame in enumerate(fus.keys()):
            fu_bitdict[funame] = fu_enable[i]
            fu_selected[funame] = fu_busy[i]

        # identify function units and create a list by fnunit so that
        # PriorityPickers can be created for selecting one of them that
        # isn't busy at the time the incoming instruction needs passing on
        by_fnunit = defaultdict(list)
        for fname, member in Function.__members__.items():
            for funame, fu in fus.items():
                fnunit = fu.fnunit.value
                if member.value & fnunit: # this FU handles this type of op
                    by_fnunit[fname].append((funame, fu)) # add by Function

        # ok now just print out the list of FUs by Function, because we can
        for fname, fu_list in by_fnunit.items():
            print ("FUs by type", fname, fu_list)

        # now create a PriorityPicker per FU-type such that only one
        # non-busy FU will be picked
        issue_pps = {}
        fu_found = Signal() # take a note if no Function Unit was available
        for fname, fu_list in by_fnunit.items():
            i_pp = PriorityPicker(len(fu_list))
            m.submodules['i_pp_%s' % fname] = i_pp
            i_l = []
            for i, (funame, fu) in enumerate(fu_list):
                # match the decoded instruction (e.do.fn_unit) against the
                # "capability" of this FU, gate that by whether that FU is
                # busy, and drop that into the PriorityPicker.
                # this will give us an output of the first available *non-busy*
                # Function Unit (Reservation Statio) capable of handling this
                # instruction.
                fnunit = fu.fnunit.value
                en_req = Signal(name="issue_en_%s" % funame, reset_less=True)
                fnmatch = (self.ireg.e.do.fn_unit & fnunit).bool()
                comb += en_req.eq(fnmatch & ~fu.busy_o &
                                    self.instr_active)
                i_l.append(en_req) # store in list for doing the Cat-trick
                # picker output, gated by enable: store in fu_bitdict
                po = Signal(name="o_issue_pick_"+funame) # picker output
                comb += po.eq(i_pp.o[i] & i_pp.en_o)
                comb += fu_bitdict[funame].eq(po)
                comb += fu_selected[funame].eq(fu.busy_o | po)
                # if we don't do this, then when there are no FUs available,
                # the "p.o_ready" signal will go back "ok we accepted this
                # instruction" which of course isn't true.
                with m.If(i_pp.en_o):
                    comb += fu_found.eq(1)
            # for each input, Cat them together and drop them into the picker
            comb += i_pp.i.eq(Cat(*i_l))

        # rdmask, which is for registers needs to come from the *main* decoder
        for funame, fu in fus.items():
            rdmask = get_rdflags(m, self.ireg.e, fu)
            comb += fu.rdmaskn.eq(~rdmask)

        # sigh - need a NOP counter
        counter = Signal(2)
        with m.If(counter != 0):
            sync += counter.eq(counter - 1)
            comb += busy_o.eq(1)

        # default to reading from incoming instruction: may be overridden
        # by copy from latch when "waiting"
        comb += self.ireg.eq(self.i)
        # always say "ready" except if overridden
        comb += self.p.o_ready.eq(1)

        with m.FSM():
            with m.State("READY"):
                with m.If(self.p.i_valid): # run only when valid
                    with m.Switch(self.ireg.e.do.insn_type):
                        # check for ATTN: halt if true
                        with m.Case(MicrOp.OP_ATTN):
                            m.d.sync += self.o.core_terminate_o.eq(1)

                        # fake NOP - this isn't really used (Issuer detects NOP)
                        with m.Case(MicrOp.OP_NOP):
                            sync += counter.eq(2)
                            comb += busy_o.eq(1)

                        with m.Default():
                            comb += self.instr_active.eq(1)
                            comb += self.p.o_ready.eq(0)
                            # connect instructions. only one enabled at a time
                            for funame, fu in fus.items():
                                do = self.des[funame]
                                enable = fu_bitdict[funame]

                                # run this FunctionUnit if enabled route op,
                                # issue, busy, read flags and mask to FU
                                with m.If(enable):
                                    # operand comes from the *local*  decoder
                                    # do not actually issue, though, if there
                                    # is a waw hazard. decoder has to still
                                    # be asserted in order to detect that, tho
                                    comb += fu.oper_i.eq_from(do)
                                    if funame == 'mmu0':
                                        # URRR this is truly dreadful.
                                        # OP_FETCH_FAILED is a "fake" op.
                                        # no instruction creates it.  OP_TRAP
                                        # uses the *main* decoder: this is
                                        # a *Satellite* decoder that reacts
                                        # on *insn_in*... not fake ops. gaah.
                                        main_op = self.ireg.e.do
                                        with m.If(main_op.insn_type ==
                                                  MicrOp.OP_FETCH_FAILED):
                                            comb += fu.oper_i.insn_type.eq(
                                                  MicrOp.OP_FETCH_FAILED)
                                            comb += fu.oper_i.fn_unit.eq(
                                                  Function.MMU)
                                    # issue when valid (and no write-hazard)
                                    comb += fu.issue_i.eq(~self.waw_hazard)
                                    # instruction ok, indicate ready
                                    comb += self.p.o_ready.eq(1)

                            if self.allow_overlap:
                                with m.If(~fu_found | self.waw_hazard):
                                    # latch copy of instruction
                                    sync += ilatch.eq(self.i)
                                    comb += self.p.o_ready.eq(1) # accept
                                    comb += busy_o.eq(1)
                                    m.next = "WAITING"

            with m.State("WAITING"):
                comb += self.instr_active.eq(1)
                comb += self.p.o_ready.eq(0)
                comb += busy_o.eq(1)
                # using copy of instruction, keep waiting until an FU is free
                comb += self.ireg.eq(ilatch)
                with m.If(fu_found): # wait for conflict to clear
                    # connect instructions. only one enabled at a time
                    for funame, fu in fus.items():
                        do = self.des[funame]
                        enable = fu_bitdict[funame]

                        # run this FunctionUnit if enabled route op,
                        # issue, busy, read flags and mask to FU
                        with m.If(enable):
                            # operand comes from the *local* decoder,
                            # which is asserted even if not issued,
                            # so that WaW-detection can check for hazards.
                            # only if the waw hazard is clear does the
                            # instruction actually get issued
                            comb += fu.oper_i.eq_from(do)
                            # issue when valid
                            comb += fu.issue_i.eq(~self.waw_hazard)
                            with m.If(~self.waw_hazard):
                                comb += self.p.o_ready.eq(1)
                                comb += busy_o.eq(0)
                                m.next = "READY"

        print ("core: overlap allowed", self.allow_overlap)
        # true when any FU is busy (including the cycle where it is perhaps
        # to be issued - because that's what fu_busy is)
        comb += any_busy_o.eq(fu_busy.bool())
        if not self.allow_overlap:
            # for simple non-overlap, if any instruction is busy, set
            # busy output for core.
            comb += busy_o.eq(any_busy_o)
        else:
            # sigh deal with a fun situation that needs to be investigated
            # and resolved
            with m.If(self.issue_conflict):
                comb += busy_o.eq(1)
            # make sure that LDST, SPR, MMU, Branch and Trap all say "busy"
            # and do not allow overlap.  these are all the ones that
            # are non-forward-progressing: exceptions etc. that otherwise
            # change CoreState for some reason (MSR, PC, SVSTATE)
            for funame, fu in fus.items():
                if (funame.lower().startswith('ldst') or
                    funame.lower().startswith('branch') or
                    funame.lower().startswith('mmu') or
                    funame.lower().startswith('spr') or
                    funame.lower().startswith('trap')):
                    with m.If(fu.busy_o):
                        comb += busy_o.eq(1)
                # for SPR pipeline pause dec/tb FSM to avoid race condition
                # TODO: really this should be much more sophisticated,
                # spot MTSPR, spot that DEC/TB is what is to be updated.
                # a job for PowerDecoder2, there
                if funame.lower().startswith('spr'):
                    with m.If(fu.busy_o #& fu.oper_i.insn_type == OP_MTSPR
                        ):
                        comb += self.pause_dec_tb.eq(1)

        # return both the function unit "enable" dict as well as the "busy".
        # the "busy-or-issued" can be passed in to the Read/Write port
        # connecters to give them permission to request access to regfiles
        return fu_bitdict, fu_selected

    def connect_rdport(self, m, fu_bitdict, fu_selected,
                                rdpickers, regfile, regname, fspec):
        comb, sync = m.d.comb, m.d.sync
        fus = self.fus.fus
        regs = self.regs

        rpidx = regname

        # select the required read port.  these are pre-defined sizes
        rfile = regs.rf[regfile.lower()]
        rport = rfile.r_ports[rpidx]
        print("read regfile", rpidx, regfile, regs.rf.keys(),
                              rfile, rfile.unary)

        # for checking if the read port has an outstanding write
        if self.make_hazard_vecs:
            wv = regs.wv[regfile.lower()]
            wvchk = wv.q_int # write-vec bit-level hazard check

        # if a hazard is detected on this read port, simply blithely block
        # every FU from reading on it.  this is complete overkill but very
        # simple for now.
        hazard_detected = Signal(name="raw_%s_%s" % (regfile, rpidx))

        fspecs = fspec
        if not isinstance(fspecs, list):
            fspecs = [fspecs]

        rdflags = []
        pplen = 0
        ppoffs = []
        for i, fspec in enumerate(fspecs):
            # get the regfile specs for this regfile port
            print ("fpsec", i, fspec, len(fspec.specs))
            name = "%s_%s_%d" % (regfile, regname, i)
            ppoffs.append(pplen) # record offset for picker
            pplen += len(fspec.specs)
            rdflag = Signal(name="rdflag_"+name, reset_less=True)
            comb += rdflag.eq(fspec.okflag)
            rdflags.append(rdflag)

        print ("pplen", pplen)

        # create a priority picker to manage this port
        rdpickers[regfile][rpidx] = rdpick = PriorityPicker(pplen)
        m.submodules["rdpick_%s_%s" % (regfile, rpidx)] = rdpick

        rens = []
        addrs = []
        wvens = []

        for i, fspec in enumerate(fspecs):
            (rf, _read, wid, fuspecs) = \
                (fspec.okflag, fspec.regport, fspec.wid, fspec.specs)
            # connect up the FU req/go signals, and the reg-read to the FU
            # and create a Read Broadcast Bus
            for pi, fuspec in enumerate(fspec.specs):
                (funame, fu, idx) = (fuspec.funame, fuspec.fu, fuspec.idx)
                pi += ppoffs[i]
                name = "%s_%s_%s_%i" % (regfile, rpidx, funame, pi)
                fu_active = fu_selected[funame]
                fu_issued = fu_bitdict[funame]

                # get (or set up) a latched copy of read register number
                # and (sigh) also the read-ok flag
                # TODO: use nmutil latchregister
                rhname = "%s_%s_%d" % (regfile, regname, i)
                rdflag = Signal(name="rdflag_%s_%s" % (funame, rhname),
                                reset_less=True)
                if rhname not in fu.rf_latches:
                    rfl = Signal(name="rdflag_latch_%s_%s" % (funame, rhname))
                    fu.rf_latches[rhname] = rfl
                    with m.If(fu.issue_i):
                        sync += rfl.eq(rdflags[i])
                else:
                    rfl = fu.rf_latches[rhname]

                # now the register port
                rname = "%s_%s_%s_%d" % (funame, regfile, regname, pi)
                read = Signal.like(_read, name="read_"+rname)
                if rname not in fu.rd_latches:
                    rdl = Signal.like(_read, name="rdlatch_"+rname)
                    fu.rd_latches[rname] = rdl
                    with m.If(fu.issue_i):
                        sync += rdl.eq(_read)
                else:
                    rdl = fu.rd_latches[rname]

                # make the read immediately available on issue cycle
                # after the read cycle, otherwies use the latched copy.
                # this captures the regport and okflag on issue
                with m.If(fu.issue_i):
                    comb += read.eq(_read)
                    comb += rdflag.eq(rdflags[i])
                with m.Else():
                    comb += read.eq(rdl)
                    comb += rdflag.eq(rfl)

                # connect request-read to picker input, and output to go-rd
                addr_en = Signal.like(read, name="addr_en_"+name)
                pick = Signal(name="pick_"+name)     # picker input
                rp = Signal(name="rp_"+name)         # picker output
                delay_pick = Signal(name="dp_"+name) # read-enable "underway"
                rhazard = Signal(name="rhaz_"+name)

                # exclude any currently-enabled read-request (mask out active)
                # entirely block anything hazarded from being picked
                comb += pick.eq(fu.rd_rel_o[idx] & fu_active & rdflag &
                                ~delay_pick & ~rhazard)
                comb += rdpick.i[pi].eq(pick)
                comb += fu.go_rd_i[idx].eq(delay_pick) # pass in *delayed* pick

                # if picked, select read-port "reg select" number to port
                comb += rp.eq(rdpick.o[pi] & rdpick.en_o)
                sync += delay_pick.eq(rp) # delayed "pick"
                comb += addr_en.eq(Mux(rp, read, 0))

                # the read-enable happens combinatorially (see mux-bus below)
                # but it results in the data coming out on a one-cycle delay.
                if rfile.unary:
                    rens.append(addr_en)
                else:
                    addrs.append(addr_en)
                    rens.append(rp)

                # use the *delayed* pick signal to put requested data onto bus
                with m.If(delay_pick):
                    # connect regfile port to input, creating fan-out Bus
                    src = fu.src_i[idx]
                    print("reg connect widths",
                          regfile, regname, pi, funame,
                          src.shape(), rport.o_data.shape())
                    # all FUs connect to same port
                    comb += src.eq(rport.o_data)

                if not self.make_hazard_vecs:
                    continue

                # read the write-hazard bitvector (wv) for any bit that is
                wvchk_en = Signal(len(wvchk), name="wv_chk_addr_en_"+name)
                issue_active = Signal(name="rd_iactive_"+name)
                # XXX combinatorial loop here
                comb += issue_active.eq(fu_active & rdflag)
                with m.If(issue_active):
                    if rfile.unary:
                        comb += wvchk_en.eq(read)
                    else:
                        comb += wvchk_en.eq(1<<read)
                # if FU is busy (which doesn't get set at the same time as
                # issue) and no hazard was detected, clear wvchk_en (i.e.
                # stop checking for hazards).  there is a loop here, but it's
                # via a DFF, so is ok. some linters may complain, but hey.
                with m.If(fu.busy_o & ~rhazard):
                        comb += wvchk_en.eq(0)

                # read-hazard is ANDed with (filtered by) what is actually
                # being requested.
                comb += rhazard.eq((wvchk & wvchk_en).bool())

                wvens.append(wvchk_en)

        # or-reduce the muxed read signals
        if rfile.unary:
            # for unary-addressed
            comb += rport.ren.eq(ortreereduce_sig(rens))
        else:
            # for binary-addressed
            comb += rport.addr.eq(ortreereduce_sig(addrs))
            comb += rport.ren.eq(Cat(*rens).bool())
            print ("binary", regfile, rpidx, rport, rport.ren, rens, addrs)

        if not self.make_hazard_vecs:
            return Const(0) # declare "no hazards"

        # enable the read bitvectors for this issued instruction
        # and return whether any write-hazard bit is set
        wvchk_and = Signal(len(wvchk), name="wv_chk_"+name)
        comb += wvchk_and.eq(wvchk & ortreereduce_sig(wvens))
        comb += hazard_detected.eq(wvchk_and.bool())
        return hazard_detected

    def connect_rdports(self, m, fu_bitdict, fu_selected):
        """connect read ports

        orders the read regspecs into a dict-of-dicts, by regfile, by
        regport name, then connects all FUs that want that regport by
        way of a PriorityPicker.
        """
        comb, sync = m.d.comb, m.d.sync
        fus = self.fus.fus
        regs = self.regs
        rd_hazard = []

        # dictionary of lists of regfile read ports
        byregfiles_rdspec = self.get_byregfiles(m, True)

        # okaay, now we need a PriorityPicker per regfile per regfile port
        # loootta pickers... peter piper picked a pack of pickled peppers...
        rdpickers = {}
        for regfile, fuspecs in byregfiles_rdspec.items():
            rdpickers[regfile] = {}

            # argh.  an experiment to merge RA and RB in the INT regfile
            # (we have too many read/write ports)
            if self.regreduce_en:
                if regfile == 'INT':
                    fuspecs['rabc'] = [fuspecs.pop('rb')]
                    fuspecs['rabc'].append(fuspecs.pop('rc'))
                    fuspecs['rabc'].append(fuspecs.pop('ra'))
                if regfile == 'FAST':
                    fuspecs['fast1'] = [fuspecs.pop('fast1')]
                    if 'fast2' in fuspecs:
                        fuspecs['fast1'].append(fuspecs.pop('fast2'))
                    if 'fast3' in fuspecs:
                        fuspecs['fast1'].append(fuspecs.pop('fast3'))

            # for each named regfile port, connect up all FUs to that port
            # also return (and collate) hazard detection)
            for (regname, fspec) in sort_fuspecs(fuspecs):
                print("connect rd", regname, fspec)
                rh = self.connect_rdport(m, fu_bitdict, fu_selected,
                                       rdpickers, regfile,
                                       regname, fspec)
                rd_hazard.append(rh)

        return Cat(*rd_hazard).bool()

    def make_hazards(self, m, regfile, rfile, wvclr, wvset,
                    funame, regname, idx,
                    addr_en, wp, fu, fu_active, wrflag, write,
                    fu_wrok):
        """make_hazards: a setter and a clearer for the regfile write ports

        setter is at issue time (using PowerDecoder2 regfile write numbers)
        clearer is at regfile write time (when FU has said what to write to)

        there is *one* unusual case here which has to be dealt with:
        when the Function Unit does *NOT* request a write to the regfile
        (has its data.ok bit CLEARED).  this is perfectly legitimate.
        and a royal pain.
        """
        comb, sync = m.d.comb, m.d.sync
        name = "%s_%s_%d" % (funame, regname, idx)

        # connect up the bitvector write hazard.  unlike the
        # regfile writeports, a ONE must be written to the corresponding
        # bit of the hazard bitvector (to indicate the existence of
        # the hazard)

        # the detection of what shall be written to is based
        # on *issue*.  it is delayed by 1 cycle so that instructions
        # "addi 5,5,0x2" do not cause combinatorial loops due to
        # fake-dependency on *themselves*.  this will totally fail
        # spectacularly when doing multi-issue
        print ("write vector (for regread)", regfile, wvset)
        wviaddr_en = Signal(len(wvset), name="wv_issue_addr_en_"+name)
        issue_active = Signal(name="iactive_"+name)
        sync += issue_active.eq(fu.issue_i & fu_active & wrflag)
        with m.If(issue_active):
            if rfile.unary:
                comb += wviaddr_en.eq(write)
            else:
                comb += wviaddr_en.eq(1<<write)

        # deal with write vector clear: this kicks in when the regfile
        # is written to, and clears the corresponding bitvector entry
        print ("write vector", regfile, wvclr)
        wvaddr_en = Signal(len(wvclr), name="wvaddr_en_"+name)
        if rfile.unary:
            comb += wvaddr_en.eq(addr_en)
        else:
            with m.If(wp):
                comb += wvaddr_en.eq(1<<addr_en)

        # XXX ASSUME that LDSTFunctionUnit always sets the data it intends to
        # this may NOT be the case when an exception occurs
        if isinstance(fu, LDSTFunctionUnit):
            return wvaddr_en, wviaddr_en

        # okaaay, this is preparation for the awkward case.
        # * latch a copy of wrflag when issue goes high.
        # * when the fu_wrok (data.ok) flag is NOT set,
        #   but the FU is done, the FU is NEVER going to write
        #   so the bitvector has to be cleared.
        latch_wrflag = Signal(name="latch_wrflag_"+name)
        with m.If(~fu.busy_o):
            sync += latch_wrflag.eq(0)
        with m.If(fu.issue_i & fu_active):
            sync += latch_wrflag.eq(wrflag)
        with m.If(fu.alu_done_o & latch_wrflag & ~fu_wrok):
            if rfile.unary:
                comb += wvaddr_en.eq(write) # addr_en gated with wp, don't use
            else:
                comb += wvaddr_en.eq(1<<addr_en) # binary addr_en not gated

        return wvaddr_en, wviaddr_en

    def connect_wrport(self, m, fu_bitdict, fu_selected,
                                wrpickers, regfile, regname, fspec):
        comb, sync = m.d.comb, m.d.sync
        fus = self.fus.fus
        regs = self.regs

        rpidx = regname

        # select the required write port.  these are pre-defined sizes
        rfile = regs.rf[regfile.lower()]
        wport = rfile.w_ports[rpidx]

        print("connect wr", regname, "unary", rfile.unary, fspec)
        print(regfile, regs.rf.keys())

        # select the write-protection hazard vector.  note that this still
        # requires to WRITE to the hazard bitvector!  read-requests need
        # to RAISE the bitvector (set it to 1), which, duh, requires a WRITE
        if self.make_hazard_vecs:
            wv = regs.wv[regfile.lower()]
            wvset = wv.s # write-vec bit-level hazard ctrl
            wvclr = wv.r # write-vec bit-level hazard ctrl
            wvchk = wv.q # write-after-write hazard check

        fspecs = fspec
        if not isinstance(fspecs, list):
            fspecs = [fspecs]

        pplen = 0
        writes = []
        ppoffs = []
        wrflags = []
        for i, fspec in enumerate(fspecs):
            # get the regfile specs for this regfile port
            (wf, _write, wid, fuspecs) = \
                (fspec.okflag, fspec.regport, fspec.wid, fspec.specs)
            print ("fpsec", i, "wrflag", wf, fspec, len(fuspecs))
            ppoffs.append(pplen) # record offset for picker
            pplen += len(fuspecs)

            name = "%s_%s_%d" % (regfile, regname, i)
            wrflag = Signal(name="wr_flag_"+name)
            if wf is not None:
                comb += wrflag.eq(wf)
            else:
                comb += wrflag.eq(0)
            wrflags.append(wrflag)

        # create a priority picker to manage this port
        wrpickers[regfile][rpidx] = wrpick = PriorityPicker(pplen)
        m.submodules["wrpick_%s_%s" % (regfile, rpidx)] = wrpick

        wsigs = []
        wens = []
        wvsets = []
        wvseten = []
        wvclren = []
        #wvens = [] - not needed: reading of writevec is permanently held hi
        addrs = []
        for i, fspec in enumerate(fspecs):
            # connect up the FU req/go signals and the reg-read to the FU
            # these are arbitrated by Data.ok signals
            (wf, _write, wid, fuspecs) = \
                (fspec.okflag, fspec.regport, fspec.wid, fspec.specs)
            for pi, fuspec in enumerate(fspec.specs):
                (funame, fu, idx) = (fuspec.funame, fuspec.fu, fuspec.idx)
                fu_requested = fu_bitdict[funame]
                pi += ppoffs[i]
                name = "%s_%s_%s_%d" % (funame, regfile, regname, idx)
                # get (or set up) a write-latched copy of write register number
                write = Signal.like(_write, name="write_"+name)
                rname = "%s_%s_%s_%d" % (funame, regfile, regname, idx)
                if rname not in fu.wr_latches:
                    wrl = Signal.like(_write, name="wrlatch_"+rname)
                    fu.wr_latches[rname] = write
                    # do not depend on fu.issue_i here, it creates a
                    # combinatorial loop on waw checking. using the FU
                    # "enable" bitdict entry for this FU is sufficient,
                    # because the PowerDecoder2 read/write nums are
                    # valid continuously when the instruction is valid
                    with m.If(fu_requested):
                        sync += wrl.eq(_write)
                        comb += write.eq(_write)
                    with m.Else():
                        comb += write.eq(wrl)
                else:
                    write = fu.wr_latches[rname]

                # write-request comes from dest.ok
                dest = fu.get_out(idx)
                fu_dest_latch = fu.get_fu_out(idx)  # latched output
                name = "%s_%s_%d" % (funame, regname, idx)
                fu_wrok = Signal(name="fu_wrok_"+name, reset_less=True)
                comb += fu_wrok.eq(dest.ok & fu.busy_o)

                # connect request-write to picker input, and output to go-wr
                fu_active = fu_selected[funame]
                pick = fu.wr.rel_o[idx] & fu_active
                comb += wrpick.i[pi].eq(pick)
                # create a single-pulse go write from the picker output
                wr_pick = Signal(name="wpick_%s_%s_%d" % (funame, regname, idx))
                comb += wr_pick.eq(wrpick.o[pi] & wrpick.en_o)
                comb += fu.go_wr_i[idx].eq(rising_edge(m, wr_pick))

                # connect the regspec write "reg select" number to this port
                # only if one FU actually requests (and is granted) the port
                # will the write-enable be activated
                wname = "waddr_en_%s_%s_%d" % (funame, regname, idx)
                addr_en = Signal.like(write, name=wname)
                wp = Signal()
                comb += wp.eq(wr_pick & wrpick.en_o)
                comb += addr_en.eq(Mux(wp, write, 0))
                if rfile.unary:
                    wens.append(addr_en)
                else:
                    addrs.append(addr_en)
                    wens.append(wp)

                # connect regfile port to input
                print("reg connect widths",
                      regfile, regname, pi, funame,
                      dest.shape(), wport.i_data.shape())
                wsigs.append(fu_dest_latch)

                # now connect up the bitvector write hazard
                if not self.make_hazard_vecs:
                    continue
                res = self.make_hazards(m, regfile, rfile, wvclr, wvset,
                                        funame, regname, idx,
                                        addr_en, wp, fu, fu_active,
                                        wrflags[i], write, fu_wrok)
                wvaddr_en, wv_issue_en = res
                wvclren.append(wvaddr_en)   # set only: no data => clear bit
                wvseten.append(wv_issue_en) # set data same as enable

                # read the write-hazard bitvector (wv) for any bit that is
                fu_requested = fu_bitdict[funame]
                wvchk_en = Signal(len(wvchk), name="waw_chk_addr_en_"+name)
                issue_active = Signal(name="waw_iactive_"+name)
                whazard = Signal(name="whaz_"+name)
                if wf is None:
                    # XXX EEK! STATE regfile (branch) does not have an
                    # write-active indicator in regspec_decode_write()
                    print ("XXX FIXME waw_iactive", issue_active,
                                                    fu_requested, wf)
                else:
                    # check bits from the incoming instruction.  note (back
                    # in connect_instruction) that the decoder is held for
                    # us to be able to do this, here... *without* issue being
                    # held HI.  we MUST NOT gate this with fu.issue_i or
                    # with fu_bitdict "enable": it would create a loop
                    comb += issue_active.eq(wf)
                with m.If(issue_active):
                    if rfile.unary:
                        comb += wvchk_en.eq(write)
                    else:
                        comb += wvchk_en.eq(1<<write)
                # if FU is busy (which doesn't get set at the same time as
                # issue) and no hazard was detected, clear wvchk_en (i.e.
                # stop checking for hazards).  there is a loop here, but it's
                # via a DFF, so is ok. some linters may complain, but hey.
                with m.If(fu.busy_o & ~whazard):
                        comb += wvchk_en.eq(0)

                # write-hazard is ANDed with (filtered by) what is actually
                # being requested.  the wvchk data is on a one-clock delay,
                # and wvchk_en comes directly from the main decoder
                comb += whazard.eq((wvchk & wvchk_en).bool())
                with m.If(whazard):
                    comb += fu._waw_hazard.eq(1)

                #wvens.append(wvchk_en)

        # here is where we create the Write Broadcast Bus. simple, eh?
        comb += wport.i_data.eq(ortreereduce_sig(wsigs))
        if rfile.unary:
            # for unary-addressed
            comb += wport.wen.eq(ortreereduce_sig(wens))
        else:
            # for binary-addressed
            comb += wport.addr.eq(ortreereduce_sig(addrs))
            comb += wport.wen.eq(ortreereduce_sig(wens))

        if not self.make_hazard_vecs:
            return [], []

        # return these here rather than set wvclr/wvset directly,
        # because there may be more than one write-port to a given
        # regfile.  example: XER has a write-port for SO, CA, and OV
        # and the *last one added* of those would overwrite the other
        # two.  solution: have connect_wrports collate all the
        # or-tree-reduced bitvector set/clear requests and drop them
        # in as a single "thing".  this can only be done because the
        # set/get is an unary bitvector.
        print ("make write-vecs", regfile, regname, wvset, wvclr)
        return (wvclren, # clear (regfile write)
                wvseten) # set (issue time)

    def connect_wrports(self, m, fu_bitdict, fu_selected):
        """connect write ports

        orders the write regspecs into a dict-of-dicts, by regfile,
        by regport name, then connects all FUs that want that regport
        by way of a PriorityPicker.

        note that the write-port wen, write-port data, and go_wr_i all need to
        be on the exact same clock cycle.  as there is a combinatorial loop bug
        at the moment, these all use sync.
        """
        comb, sync = m.d.comb, m.d.sync
        fus = self.fus.fus
        regs = self.regs
        # dictionary of lists of regfile write ports
        byregfiles_wrspec = self.get_byregfiles(m, False)

        # same for write ports.
        # BLECH!  complex code-duplication! BLECH!
        wrpickers = {}
        wvclrers = defaultdict(list)
        wvseters = defaultdict(list)
        for regfile, fuspecs in byregfiles_wrspec.items():
            wrpickers[regfile] = {}

            if self.regreduce_en:
                # argh, more port-merging
                if regfile == 'INT':
                    fuspecs['o'] = [fuspecs.pop('o')]
                    fuspecs['o'].append(fuspecs.pop('o1'))
                if regfile == 'FAST':
                    fuspecs['fast1'] = [fuspecs.pop('fast1')]
                    if 'fast2' in fuspecs:
                        fuspecs['fast1'].append(fuspecs.pop('fast2'))
                    if 'fast3' in fuspecs:
                        fuspecs['fast1'].append(fuspecs.pop('fast3'))

            # collate these and record them by regfile because there
            # are sometimes more write-ports per regfile
            for (regname, fspec) in sort_fuspecs(fuspecs):
                wvclren, wvseten = self.connect_wrport(m,
                                        fu_bitdict, fu_selected,
                                        wrpickers,
                                        regfile, regname, fspec)
                wvclrers[regfile.lower()] += wvclren
                wvseters[regfile.lower()] += wvseten

        if not self.make_hazard_vecs:
            return

        # for write-vectors: reduce the clr-ers and set-ers down to
        # a single set of bits.  otherwise if there are two write
        # ports (on some regfiles), the last one doing comb += on
        # the reg.wv[regfile] instance "wins" (and all others are ignored,
        # whoops).  if there was only one write-port per wv regfile this would
        # not be an issue.
        for regfile in wvclrers.keys():
            wv = regs.wv[regfile]
            wvset = wv.s # write-vec bit-level hazard ctrl
            wvclr = wv.r # write-vec bit-level hazard ctrl
            wvclren = wvclrers[regfile]
            wvseten = wvseters[regfile]
            comb += wvclr.eq(ortreereduce_sig(wvclren)) # clear (regfile write)
            comb += wvset.eq(ortreereduce_sig(wvseten)) # set (issue time)

    def get_byregfiles(self, m, readmode):

        mode = "read" if readmode else "write"
        regs = self.regs
        fus = self.fus.fus
        e = self.ireg.e # decoded instruction to execute

        # dictionary of dictionaries of lists/tuples of regfile ports.
        # first key: regfile.  second key: regfile port name
        byregfiles_spec = defaultdict(dict)

        for (funame, fu) in fus.items():
            # create in each FU a receptacle for the read/write register
            # hazard numbers (and okflags for read).  to be latched in
            # connect_rd/write_ports
            if readmode:
                fu.rd_latches = {} # read reg number latches
                fu.rf_latches = {} # read flag latches
            else:
                fu.wr_latches = {}

            # construct regfile specs: read uses inspec, write outspec
            print("%s ports for %s" % (mode, funame))
            for idx in range(fu.n_src if readmode else fu.n_dst):
                (regfile, regname, wid) = fu.get_io_spec(readmode, idx)
                print("    %d %s %s %s" % (idx, regfile, regname, str(wid)))

                # the PowerDecoder2 (main one, not the satellites) contains
                # the decoded regfile numbers. obtain these now
                decinfo = regspec_decode(m, readmode, e, regfile, regname)
                okflag, regport = decinfo.okflag, decinfo.regport

                # construct the dictionary of regspec information by regfile
                if regname not in byregfiles_spec[regfile]:
                    byregfiles_spec[regfile][regname] = \
                        ByRegSpec(okflag, regport, wid, [])

                # here we start to create "lanes" where each Function Unit
                # requiring access to a given [single-contended resource]
                # regfile port is appended to a list, so that PriorityPickers
                # can be created to give uncontested access to it
                fuspec = FUSpec(funame, fu, idx)
                byregfiles_spec[regfile][regname].specs.append(fuspec)

        # ok just print that all out, for convenience
        for regfile, fuspecs in byregfiles_spec.items():
            print("regfile %s ports:" % mode, regfile)
            for regname, fspec in fuspecs.items():
                [okflag, regport, wid, fuspecs] = fspec
                print("  rf %s port %s lane: %s" % (mode, regfile, regname))
                print("  %s" % regname, wid, okflag, regport)
                for (funame, fu, idx) in fuspecs:
                    fusig = fu.src_i[idx] if readmode else fu.dest[idx]
                    print("    ", funame, fu.__class__.__name__, idx, fusig)
                    print()

        return byregfiles_spec

    def __iter__(self):
        yield from self.fus.ports()
        yield from self.i.e.ports()
        yield from self.l0.ports()
        # TODO: regs

    def ports(self):
        return list(self)


if __name__ == '__main__':
    pspec = TestMemPspec(ldst_ifacetype='testpi',
                         imem_ifacetype='',
                         addr_wid=64,
                         allow_overlap=True,
                         mask_wid=8,
                         reg_wid=64)
    dut = NonProductionCore(pspec)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_core.il", "w") as f:
        f.write(vl)
