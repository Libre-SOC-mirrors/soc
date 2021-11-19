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
"""

from nmigen import Elaboratable, Module, Signal, ResetSignal, Cat, Mux
from nmigen.cli import rtlil

from openpower.decoder.power_decoder2 import PowerDecodeSubset
from openpower.decoder.power_regspec_map import regspec_decode_read
from openpower.decoder.power_regspec_map import regspec_decode_write
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

from collections import defaultdict
import operator

from nmutil.util import rising_edge


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
        self.make_hazard_vecs = True
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
        print ("core pspec", pspec.ldst_ifacetype)
        print ("core mmu", mmu)
        if mmu is not None:
            print ("core lsmem.lsi", l0.cmpi.lsmem.lsi)
            mmu.alu.set_ldst_interface(l0.cmpi.lsmem.lsi)

        # register files (yes plural)
        self.regs = RegFiles(pspec, make_hazard_vecs=self.make_hazard_vecs)

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

        for funame, fu in self.fus.fus.items():
            f_name = fu.fnunit.name
            fnunit = fu.fnunit.value
            opkls = fu.opsubsetkls
            if f_name == 'TRAP':
                # TRAP decoder is the *main* decoder
                self.trapunit = funame
                continue
            self.decoders[funame] = PowerDecodeSubset(None, opkls, f_name,
                                                      final=True,
                                                      state=self.ireg.state,
                                            svp64_en=self.svp64_en,
                                            regreduce_en=self.regreduce_en)
            self.des[funame] = self.decoders[funame].do

        # share the SPR decoder with the MMU if it exists
        if "mmu0" in self.decoders:
            self.decoders["mmu0"].mmu0_spr_dec = self.decoders["spr0"]

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
            setattr(m.submodules, "dec_%s" % v.fn_name, v)
            comb += v.dec.raw_opcode_in.eq(self.ireg.raw_insn_i)
            comb += v.dec.bigendian.eq(self.ireg.bigendian_i)
            # sigh due to SVP64 RA_OR_ZERO detection connect these too
            comb += v.sv_a_nz.eq(self.ireg.sv_a_nz)
            if self.svp64_en:
                comb += v.pred_sm.eq(self.ireg.sv_pred_sm)
                comb += v.pred_dm.eq(self.ireg.sv_pred_dm)
                if k != self.trapunit:
                    comb += v.sv_rm.eq(self.ireg.sv_rm) # pass through SVP64 RM
                    comb += v.is_svp64_mode.eq(self.ireg.is_svp64_mode)
                    # only the LDST PowerDecodeSubset *actually* needs to
                    # know to use the alternative decoder.  this is all
                    # a terrible hack
                    if k.lower().startswith("ldst"):
                        comb += v.use_svp64_ldst_dec.eq(
                                        self.ireg.use_svp64_ldst_dec)

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
                with m.If(~self.issue_conflict & i_pp.en_o):
                    comb += fu_found.eq(1)
            # for each input, Cat them together and drop them into the picker
            comb += i_pp.i.eq(Cat(*i_l))

        # rdmask, which is for registers needs to come from the *main* decoder
        for funame, fu in fus.items():
            rdmask = get_rdflags(self.ireg.e, fu)
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

        l_issue_conflict = Signal()

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
                                with m.If(enable & ~self.issue_conflict):
                                    # operand comes from the *local*  decoder
                                    comb += fu.oper_i.eq_from(do)
                                    comb += fu.issue_i.eq(1) # issue when valid
                                    # instruction ok, indicate ready
                                    comb += self.p.o_ready.eq(1)
                                    comb += busy_o.eq(1)

                            if self.allow_overlap:
                                with m.If(~fu_found):
                                    comb += self.instr_active.eq(1)
                                    # latch copy of instruction
                                    sync += ilatch.eq(self.i)
                                    sync += l_issue_conflict.eq(
                                                      self.issue_conflict)
                                    comb += self.p.o_ready.eq(1) # accept
                                    comb += busy_o.eq(1)
                                    m.next = "WAITING"

            with m.State("WAITING"):
                comb += self.instr_active.eq(1)
                with m.If(fu_found):
                    sync += l_issue_conflict.eq(0)
                comb += self.p.o_ready.eq(0)
                comb += busy_o.eq(1)
                # using copy of instruction, keep waiting until an FU is free
                comb += self.ireg.eq(ilatch)
                with m.If(~l_issue_conflict): # wait for conflict to clear
                    # connect instructions. only one enabled at a time
                    for funame, fu in fus.items():
                        do = self.des[funame]
                        enable = fu_bitdict[funame]

                        # run this FunctionUnit if enabled route op,
                        # issue, busy, read flags and mask to FU
                        with m.If(enable):
                            # operand comes from the *local*  decoder
                            comb += fu.oper_i.eq_from(do)
                            comb += fu.issue_i.eq(1) # issue when valid
                            comb += self.p.o_ready.eq(1)
                            comb += busy_o.eq(0)
                            m.next = "READY"

        print ("core: overlap allowed", self.allow_overlap)
        if not self.allow_overlap:
            # for simple non-overlap, if any instruction is busy, set
            # busy output for core.
            busys = map(lambda fu: fu.busy_o, fus.values())
            comb += busy_o.eq(Cat(*busys).bool())

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
            wvchk = wv.r_ports["issue"] # write-vec bit-level hazard check

        fspecs = fspec
        if not isinstance(fspecs, list):
            fspecs = [fspecs]

        rdflags = []
        pplen = 0
        ppoffs = []
        for i, fspec in enumerate(fspecs):
            # get the regfile specs for this regfile port
            (rf, wf, read, write, wid, fuspec) = fspec
            print ("fpsec", i, fspec, len(fuspec))
            ppoffs.append(pplen) # record offset for picker
            pplen += len(fuspec)
            name = "rdflag_%s_%s_%d" % (regfile, regname, i)
            rdflag = Signal(name=name, reset_less=True)
            comb += rdflag.eq(rf)
            rdflags.append(rdflag)

        print ("pplen", pplen)

        # create a priority picker to manage this port
        rdpickers[regfile][rpidx] = rdpick = PriorityPicker(pplen)
        setattr(m.submodules, "rdpick_%s_%s" % (regfile, rpidx), rdpick)

        rens = []
        addrs = []
        wvens = []

        for i, fspec in enumerate(fspecs):
            (rf, wf, _read, _write, wid, fuspec) = fspec
            # connect up the FU req/go signals, and the reg-read to the FU
            # and create a Read Broadcast Bus
            for pi, (funame, fu, idx) in enumerate(fuspec):
                pi += ppoffs[i]
                name = "%s_%s_%s_%i" % (regfile, rpidx, funame, pi)
                fu_active = fu_selected[funame]

                # get (or set up) a latched copy of read register number
                rname = "%s_%s_%s_%d" % (funame, regfile, regname, pi)
                read = Signal.like(_read, name="read_"+name)
                if rname not in fu.rd_latches:
                    rdl = Signal.like(_read, name="rdlatch_"+rname)
                    fu.rd_latches[rname] = rdl
                    with m.If(fu.issue_i):
                        sync += rdl.eq(_read)
                else:
                    rdl = fu.rd_latches[rname]
                # latch to make the read immediately available on issue cycle
                # after the read cycle, use the latched copy
                with m.If(fu.issue_i):
                    comb += read.eq(_read)
                with m.Else():
                    comb += read.eq(rdl)

                # connect request-read to picker input, and output to go-rd
                addr_en = Signal.like(read, name="addr_en_"+name)
                pick = Signal(name="pick_"+name)     # picker input
                rp = Signal(name="rp_"+name)         # picker output
                delay_pick = Signal(name="dp_"+name) # read-enable "underway"

                # exclude any currently-enabled read-request (mask out active)
                comb += pick.eq(fu.rd_rel_o[idx] & fu_active & rdflags[i] &
                                ~delay_pick)
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
                wvchk_en = Signal(len(wvchk.ren), name="wv_chk_addr_en_"+name)
                issue_active = Signal(name="rd_iactive_"+name)
                # XXX combinatorial loop here
                comb += issue_active.eq(self.instr_active & rf)
                with m.If(issue_active):
                    if rfile.unary:
                        comb += wvchk_en.eq(read)
                    else:
                        comb += wvchk_en.eq(1<<read)
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
        comb += wvchk.ren.eq(ortreereduce_sig(wvens))
        hazard_detected = Signal(name="raw_%s_%s" % (regfile, rpidx))
        comb += hazard_detected.eq(wvchk.o_data.bool())
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
        byregfiles_rd, byregfiles_rdspec = self.get_byregfiles(True)

        # okaay, now we need a PriorityPicker per regfile per regfile port
        # loootta pickers... peter piper picked a pack of pickled peppers...
        rdpickers = {}
        for regfile, spec in byregfiles_rd.items():
            fuspecs = byregfiles_rdspec[regfile]
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
        # on *issue*
        print ("write vector (for regread)", regfile, wvset)
        wviaddr_en = Signal(len(wvset.wen), name="wv_issue_addr_en_"+name)
        issue_active = Signal(name="iactive_"+name)
        comb += issue_active.eq(fu.issue_i & fu_active & wrflag)
        with m.If(issue_active):
            if rfile.unary:
                comb += wviaddr_en.eq(write)
            else:
                comb += wviaddr_en.eq(1<<write)

        # deal with write vector clear: this kicks in when the regfile
        # is written to, and clears the corresponding bitvector entry
        print ("write vector", regfile, wvclr)
        wvaddr_en = Signal(len(wvclr.wen), name="wvaddr_en_"+name)
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
            wvset = wv.w_ports["set"] # write-vec bit-level hazard ctrl
            wvclr = wv.w_ports["clr"] # write-vec bit-level hazard ctrl

        fspecs = fspec
        if not isinstance(fspecs, list):
            fspecs = [fspecs]

        pplen = 0
        writes = []
        ppoffs = []
        rdflags = []
        wrflags = []
        for i, fspec in enumerate(fspecs):
            # get the regfile specs for this regfile port
            (rf, wf, read, write, wid, fuspec) = fspec
            print ("fpsec", i, "wrflag", wf, fspec, len(fuspec))
            ppoffs.append(pplen) # record offset for picker
            pplen += len(fuspec)

            name = "%s_%s_%d" % (regfile, regname, i)
            rdflag = Signal(name="rd_flag_"+name)
            wrflag = Signal(name="wr_flag_"+name)
            if rf is not None:
                comb += rdflag.eq(rf)
            else:
                comb += rdflag.eq(0)
            if wf is not None:
                comb += wrflag.eq(wf)
            else:
                comb += wrflag.eq(0)
            rdflags.append(rdflag)
            wrflags.append(wrflag)

        # create a priority picker to manage this port
        wrpickers[regfile][rpidx] = wrpick = PriorityPicker(pplen)
        setattr(m.submodules, "wrpick_%s_%s" % (regfile, rpidx), wrpick)

        wsigs = []
        wens = []
        wvsets = []
        wvseten = []
        wvclren = []
        addrs = []
        for i, fspec in enumerate(fspecs):
            # connect up the FU req/go signals and the reg-read to the FU
            # these are arbitrated by Data.ok signals
            (rf, wf, read, _write, wid, fuspec) = fspec
            for pi, (funame, fu, idx) in enumerate(fuspec):
                pi += ppoffs[i]
                name = "%s_%s_%s_%d" % (funame, regfile, regname, idx)
                # get (or set up) a write-latched copy of write register number
                write = Signal.like(_write, name="write_"+name)
                rname = "%s_%s_%s" % (funame, regfile, regname)
                if rname not in fu.wr_latches:
                    wrl = Signal.like(_write, name="wrlatch_"+rname)
                    fu.wr_latches[rname] = write
                    with m.If(fu.issue_i):
                        sync += wrl.eq(_write)
                        comb += write.eq(_write)
                    with m.Else():
                        comb += write.eq(wrl)
                else:
                    write = fu.wr_latches[rname]

                # write-request comes from dest.ok
                dest = fu.get_out(idx)
                fu_dest_latch = fu.get_fu_out(idx)  # latched output
                name = "fu_wrok_%s_%s_%d" % (funame, regname, idx)
                fu_wrok = Signal(name=name, reset_less=True)
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
                wvsets.append(wv_issue_en)  # because enable needs a 1

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
            return

        # for write-vectors
        comb += wvclr.wen.eq(ortreereduce_sig(wvclren)) # clear (regfile write)
        comb += wvset.wen.eq(ortreereduce_sig(wvseten)) # set (issue time)
        comb += wvset.i_data.eq(ortreereduce_sig(wvsets))

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
        byregfiles_wr, byregfiles_wrspec = self.get_byregfiles(False)

        # same for write ports.
        # BLECH!  complex code-duplication! BLECH!
        wrpickers = {}
        for regfile, spec in byregfiles_wr.items():
            fuspecs = byregfiles_wrspec[regfile]
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

            for (regname, fspec) in sort_fuspecs(fuspecs):
                self.connect_wrport(m, fu_bitdict, fu_selected, wrpickers,
                                        regfile, regname, fspec)

    def get_byregfiles(self, readmode):

        mode = "read" if readmode else "write"
        regs = self.regs
        fus = self.fus.fus
        e = self.ireg.e # decoded instruction to execute

        # dictionary of dictionaries of lists of regfile ports.
        # first key: regfile.  second key: regfile port name
        byregfiles = defaultdict(dict)
        byregfiles_spec = defaultdict(dict)

        for (funame, fu) in fus.items():
            # create in each FU a receptacle for the read/write register
            # hazard numbers.  to be latched in connect_rd/write_ports
            # XXX better that this is moved into the actual FUs, but
            # the issue there is that this function is actually better
            # suited at the moment
            if readmode:
                fu.rd_latches = {}
            else:
                fu.wr_latches = {}

            print("%s ports for %s" % (mode, funame))
            for idx in range(fu.n_src if readmode else fu.n_dst):
                # construct regfile specs: read uses inspec, write outspec
                if readmode:
                    (regfile, regname, wid) = fu.get_in_spec(idx)
                else:
                    (regfile, regname, wid) = fu.get_out_spec(idx)
                print("    %d %s %s %s" % (idx, regfile, regname, str(wid)))

                # the PowerDecoder2 (main one, not the satellites) contains
                # the decoded regfile numbers. obtain these now
                if readmode:
                    rdflag, read = regspec_decode_read(e, regfile, regname)
                    wrport, write = None, None
                else:
                    rdflag, read = None, None
                    wrport, write = regspec_decode_write(e, regfile, regname)

                # construct the dictionary of regspec information by regfile
                if regname not in byregfiles_spec[regfile]:
                    byregfiles_spec[regfile][regname] = \
                        (rdflag, wrport, read, write, wid, [])
                # here we start to create "lanes"
                if idx not in byregfiles[regfile]:
                    byregfiles[regfile][idx] = []
                fuspec = (funame, fu, idx)
                byregfiles[regfile][idx].append(fuspec)
                byregfiles_spec[regfile][regname][5].append(fuspec)

                continue
                # append a latch Signal to the FU's list of latches
                rname = "%s_%s" % (regfile, regname)
                if readmode:
                    if rname not in fu.rd_latches:
                        rdl = Signal.like(read, name="rdlatch_"+rname)
                        fu.rd_latches[rname] = rdl
                else:
                    if rname not in fu.wr_latches:
                        wrl = Signal.like(write, name="wrlatch_"+rname)
                        fu.wr_latches[rname] = wrl

        # ok just print that all out, for convenience
        for regfile, spec in byregfiles.items():
            print("regfile %s ports:" % mode, regfile)
            fuspecs = byregfiles_spec[regfile]
            for regname, fspec in fuspecs.items():
                [rdflag, wrflag, read, write, wid, fuspec] = fspec
                print("  rf %s port %s lane: %s" % (mode, regfile, regname))
                print("  %s" % regname, wid, read, write, rdflag, wrflag)
                for (funame, fu, idx) in fuspec:
                    fusig = fu.src_i[idx] if readmode else fu.dest[idx]
                    print("    ", funame, fu.__class__.__name__, idx, fusig)
                    print()

        return byregfiles, byregfiles_spec

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
                         addr_wid=48,
                         mask_wid=8,
                         reg_wid=64)
    dut = NonProductionCore(pspec)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_core.il", "w") as f:
        f.write(vl)
