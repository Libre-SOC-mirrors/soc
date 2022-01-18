# POWER9 Register Files
"""POWER9 regfiles

Defines the following register files:

    * INT regfile   - 32x 64-bit
    * SPR regfile   - 110x 64-bit
    * CR regfile    - CR0-7
    * XER regfile   - XER.so, XER.ca/ca32, XER.ov/ov32
    * FAST regfile  - CTR, LR, TAR, SRR1, SRR2
    * STATE regfile  - PC, MSR, (SimpleV VL later)

Note: this should NOT have name conventions hard-coded (dedicated ports per
regname).  However it is convenient for now.

Links:

* https://bugs.libre-soc.org/show_bug.cgi?id=345
* https://bugs.libre-soc.org/show_bug.cgi?id=351
* https://libre-soc.org/3d_gpu/architecture/regfile/
* https://libre-soc.org/openpower/isatables/sprs.csv
* https://libre-soc.org/openpower/sv/sprs/ (SVSTATE)
"""

# TODO

from soc.regfile.regfile import RegFile, RegFileArray, RegFileMem
from soc.regfile.virtual_port import VirtualRegPort
from openpower.decoder.power_enums import SPRfull, SPRreduced

# XXX MAKE DAMN SURE TO KEEP THESE UP-TO-DATE if changing/adding regs
from openpower.consts import StateRegsEnum, XERRegsEnum, FastRegsEnum

from nmigen import Module
from nmigen.cli import rtlil
from nmutil.latch import SRLatch


def create_ports(rf, wr_spec, rd_spec):
    """create_ports: creates register file ports based on requested specs
    """
    rf.r_ports, rf.w_ports = {}, {}
    # create read ports based on read specs
    for key, name in rd_spec.items():
        if hasattr(rf, name): # some regfiles already have a port
            rf.r_ports[key] = getattr(rf, name)
        else:
            rf.r_ports[key] = rf.read_port(name)
    # create write ports based on write specs
    for key, name in wr_spec.items():
        if hasattr(rf, name): # some regfiles already have a port
            rf.w_ports[key] = getattr(rf, name)
        else:
            rf.w_ports[key] = rf.write_port(name)


# "State" Regfile
class StateRegs(RegFileArray, StateRegsEnum):
    """StateRegs

    State regfile  - PC, MSR, SVSTATE (for SimpleV)

    * QTY 3of 64-bit registers
    * 4R3W
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)

    Note: d_wr1 d_rd1 are for use by the decoder, to get at the PC.
    will probably have to also add one so it can get at the MSR as well.
    (d_rd2)

    """
    def __init__(self, svp64_en=False, regreduce_en=False, resets=None):
        super().__init__(64, StateRegsEnum.N_REGS, resets=resets)
        wr_spec, rd_spec = self.get_port_specs()
        create_ports(self, wr_spec, rd_spec)

    def get_port_specs(self):
        w_port_spec = { # these 3 allow writing state by Function Units
                        # strictly speaking this should not be allowed,
                        # the information should be passed back to Issuer
                        # to work out what to do
                        'nia': "nia",
                        'msr': "msr",
                        'svstate': "svstate",
                        # these 3 allow writing state by Issuer
                        'sv': "sv", # writing SVSTATE
                        'd_wr1': "d_wr1", # writing PC
                        'd_wr2': "d_wr2"} # writing MSR
        r_port_spec = { # these are for reading state by Issuer but
                        # the FUs do not read them: they are passed in
                        # because of multi-issue / pipelining / etc.
                        # the state could be totally different and is
                        # only known *at* issue time, *by* the issuer
                        'cia': "cia", # reading PC (issuer)
                        'msr': "msr", # reading MSR (issuer)
                        'sv': "sv", # reading SV (issuer)
                        }
        return w_port_spec, r_port_spec


# Integer Regfile
class IntRegs(RegFileMem): #class IntRegs(RegFileArray):
    """IntRegs

    * QTY 32of 64-bit registers
    * 3R2W
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)
    """
    def __init__(self, svp64_en=False, regreduce_en=False):
        super().__init__(64, 32, fwd_bus_mode=False)
        self.svp64_en = svp64_en
        self.regreduce_en = regreduce_en
        wr_spec, rd_spec = self.get_port_specs()
        create_ports(self, wr_spec, rd_spec)

    def get_port_specs(self):
        w_port_spec = {'o': "dest1",
                        }
        r_port_spec = { 'dmi': "dmi" # needed for Debug (DMI)
                      }
        if self.svp64_en:
            r_port_spec['pred'] = "pred" # for predicate mask
        if not self.regreduce_en:
            w_port_spec['o1'] = "dest2" # (LD/ST update)
            r_port_spec['ra'] = "src1"
            r_port_spec['rb'] = "src2"
            r_port_spec['rc'] = "src3"
        else:
            r_port_spec['rabc'] = "src1"
        return w_port_spec, r_port_spec


# Fast SPRs Regfile
class FastRegs(RegFileMem, FastRegsEnum): #RegFileArray):
    """FastRegs

    FAST regfile  - CTR, LR, TAR, SRR1, SRR2, XER, TB, DEC, SVSRR0

    * QTY 6of 64-bit registers
    * 3R2W
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)

    Note: r/w issue are used by issuer to increment/decrement TB/DEC.
    """
    def __init__(self, svp64_en=False, regreduce_en=False):
        super().__init__(64, FastRegsEnum.N_REGS, fwd_bus_mode=False)
        self.svp64_en = svp64_en
        self.regreduce_en = regreduce_en
        wr_spec, rd_spec = self.get_port_specs()
        create_ports(self, wr_spec, rd_spec)

    def get_port_specs(self):
        w_port_spec = {'fast1': "dest1",
                       'issue': "issue", # writing DEC/TB
                       }
        r_port_spec = {'fast1': "src1",
                       'issue': "issue", # reading DEC/TB
                        'dmi': "dmi" # needed for Debug (DMI)
                        }
        if not self.regreduce_en:
            r_port_spec['fast2'] = "src2"
            r_port_spec['fast3'] = "src3"
            w_port_spec['fast2'] = "dest2"
            w_port_spec['fast3'] = "dest3"

        return w_port_spec, r_port_spec


# CR Regfile
class CRRegs(VirtualRegPort):
    """Condition Code Registers (CR0-7)

    * QTY 8of 8-bit registers
    * 3R1W 4-bit-wide with additional 1R1W for the "full" 32-bit width
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)
    """
    def __init__(self, svp64_en=False, regreduce_en=False):
        super().__init__(32, 8, rd2=True)
        self.svp64_en = svp64_en
        self.regreduce_en = regreduce_en
        wr_spec, rd_spec = self.get_port_specs()
        create_ports(self, wr_spec, rd_spec)

    def get_port_specs(self):
        w_port_spec = {'full_cr': "full_wr", # 32-bit (masked, 8-en lines)
                        'cr_a': "dest1", # 4-bit, unary-indexed
                        'cr_b': "dest2"} # 4-bit, unary-indexed
        r_port_spec = {'full_cr': "full_rd", # 32-bit (masked, 8-en lines)
                        'full_cr_dbg': "full_rd2", # for DMI
                        'cr_a': "src1",
                        'cr_b': "src2",
                        'cr_c': "src3"}
        if self.svp64_en:
            r_port_spec['cr_pred'] = "cr_pred" # for predicate

        return w_port_spec, r_port_spec


# XER Regfile
class XERRegs(VirtualRegPort, XERRegsEnum):
    """XER Registers (SO, CA/CA32, OV/OV32)

    * QTY 3of 2-bit registers
    * 3R3W 2-bit-wide with additional 1R1W for the "full" 6-bit width
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)
    """
    SO=0 # this is actually 2-bit but we ignore 1 bit of it
    CA=1 # CA and CA32
    OV=2 # OV and OV32
    def __init__(self, svp64_en=False, regreduce_en=False):
        super().__init__(6, XERRegsEnum.N_REGS)
        self.svp64_en = svp64_en
        self.regreduce_en = regreduce_en
        wr_spec, rd_spec = self.get_port_specs()
        create_ports(self, wr_spec, rd_spec)

    def get_port_specs(self):
        w_port_spec = {'full_xer': "full_wr", # 6-bit (masked, 3-en lines)
                        'xer_so': "dest1",
                        'xer_ca': "dest2",
                        'xer_ov': "dest3"}
        r_port_spec = {'full_xer': "full_rd", # 6-bit (masked, 3-en lines)
                        'xer_so': "src1",
                        'xer_ca': "src2",
                        'xer_ov': "src3"}
        return w_port_spec, r_port_spec


# SPR Regfile
class SPRRegs(RegFileMem):
    """SPRRegs

    * QTY len(SPRs) 64-bit registers
    * 1R1W
    * binary-indexed but REQUIRES MAPPING
    * write-through capability (read on same cycle as write)
    """
    def __init__(self, svp64_en=False, regreduce_en=False):
        if regreduce_en:
            n_sprs = len(SPRreduced)
        else:
            n_sprs = len(SPRfull)
        super().__init__(width=64, depth=n_sprs,
                         fwd_bus_mode=False)
        self.svp64_en = svp64_en
        self.regreduce_en = regreduce_en
        wr_spec, rd_spec = self.get_port_specs()
        create_ports(self, wr_spec, rd_spec)

    def get_port_specs(self):
        w_port_spec = {'spr1': "spr1"}
        r_port_spec = {'spr1': "spr1"}
        return w_port_spec, r_port_spec


# class containing all regfiles: int, cr, xer, fast, spr
class RegFiles:
    # Factory style classes
    regkls = [('int', IntRegs),
              ('cr', CRRegs),
              ('xer', XERRegs),
              ('fast', FastRegs),
              ('state', StateRegs),
              ('spr', SPRRegs),]
    def __init__(self, pspec, make_hazard_vecs=False,
                      state_resets=None): # state file reset values
        # test is SVP64 is to be enabled
        svp64_en = hasattr(pspec, "svp64") and (pspec.svp64 == True)

        # and regfile port reduction
        regreduce_en = hasattr(pspec, "regreduce") and \
                      (pspec.regreduce == True)

        self.rf = {} # register file dict
        # create regfiles here, Factory style
        for (name, kls) in RegFiles.regkls:
            kwargs = {'svp64_en': svp64_en, 'regreduce_en': regreduce_en}
            if name == 'state':
                kwargs['resets'] = state_resets
            rf = self.rf[name] = kls(**kwargs)
            # also add these as instances, self.state, self.fast, self.cr etc.
            setattr(self, name, rf)

        self.rv, self.wv = {}, {}
        if make_hazard_vecs:
            # create a read-hazard and write-hazard vectors for this regfile
            self.wv = self.make_vecs("wr") # global write vectors
            self.rv = self.make_vecs("rd") # global read vectors

    def make_vecs(self, name):
        vec = {}
        # create regfiles here, Factory style
        for (name, kls) in RegFiles.regkls:
            rf = self.rf[name]
            vec[name] = self.make_hazard_vec(rf, name)
        return vec

    def make_hazard_vec(self, rf, name):
        if isinstance(rf, VirtualRegPort):
            vec = SRLatch(sync=False, llen=rf.nregs, name=name)
        else:
            vec = SRLatch(sync=False, llen=rf.depth, name=name)
        return vec

    def elaborate_into(self, m, platform):
        for (name, rf) in self.rf.items():
            setattr(m.submodules, name, rf)
        for (name, rv) in self.rv.items():
            setattr(m.submodules, "rv_"+name, rv)
        for (name, wv) in self.wv.items():
            setattr(m.submodules, "wv_"+name, wv)
        return m

if __name__ == '__main__':
    m = Module()
    from soc.config.test.test_loadstore import TestMemPspec
    pspec = TestMemPspec()
    rf = RegFiles(pspec, make_hazard_vecs=True)
    rf.elaborate_into(m, None)
    vl = rtlil.convert(m)
    with open("test_regfiles.il", "w") as f:
        f.write(vl)

