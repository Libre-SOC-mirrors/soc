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
"""

# TODO

from soc.regfile.regfile import RegFile, RegFileArray
from soc.regfile.virtual_port import VirtualRegPort
from soc.decoder.power_enums import SPR
from nmigen import Memory, Elaboratable


# "State" Regfile
class StateRegs(RegFileArray):
    """StateRegs

    State regfile  - PC, MSR and later SimpleV VL

    * QTY 2of 64-bit registers
    * 3R2W
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)

    Note: d_wr1 d_rd1 are for use by the decoder, to get at the PC.
    will probably have to also add one so it can get at the MSR as well.
    (d_rd2)
    """
    PC = 0
    MSR = 1
    def __init__(self):
        super().__init__(64, 2)
        self.w_ports = {'nia': self.write_port("nia"),
                        'msr': self.write_port("msr"),
                        'd_wr1': self.write_port("d_wr1")} # writing PC (issuer)
        self.r_ports = {'cia': self.read_port("cia"), # reading PC (issuer)
                        'msr': self.read_port("msr"), # reading MSR (issuer)
                        }


# Integer Regfile
class IntRegs(RegFileArray):
    """IntRegs

    * QTY 32of 64-bit registers
    * 3R2W
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)
    """
    def __init__(self):
        super().__init__(64, 32)
        self.w_ports = {'o': self.write_port("dest1"),
                        #'o1': self.write_port("dest2") # for now (LD/ST update)
                        }
        self.r_ports = {'rabc': self.read_port("src1"),
                        #'rbc': self.read_port("src3"),
                        'dmi': self.read_port("dmi")} # needed for Debug (DMI)


# Fast SPRs Regfile
class FastRegs(RegFileArray):
    """FastRegs

    FAST regfile  - CTR, LR, TAR, SRR1, SRR2

    * QTY 5of 64-bit registers
    * 2R1W
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)
    """
    CTR = 0
    LR = 1
    TAR = 2
    SRR0 = 3
    SRR1 = 4
    def __init__(self):
        super().__init__(64, 5)
        self.w_ports = {'fast1': self.write_port("dest3"),
                       }
        self.r_ports = {'fast1': self.read_port("src1"),
                        }


# CR Regfile
class CRRegs(VirtualRegPort):
    """Condition Code Registers (CR0-7)

    * QTY 8of 8-bit registers
    * 3R1W 4-bit-wide with additional 1R1W for the "full" 32-bit width
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)
    """
    def __init__(self):
        super().__init__(32, 8)
        self.w_ports = {'full_cr': self.full_wr, # 32-bit (masked, 8-en lines)
                        'cr_a': self.write_port("dest1"), # 4-bit, unary-indexed
                        'cr_b': self.write_port("dest2")} # 4-bit, unary-indexed
        self.r_ports = {'full_cr': self.full_rd, # 32-bit (masked, 8-en lines)
                        'cr_a': self.read_port("src1"),
                        'cr_b': self.read_port("src2"),
                        'cr_c': self.read_port("src3")}


# XER Regfile
class XERRegs(VirtualRegPort):
    """XER Registers (SO, CA/CA32, OV/OV32)

    * QTY 3of 2-bit registers
    * 3R3W 2-bit-wide with additional 1R1W for the "full" 6-bit width
    * Array-based unary-indexed (not binary-indexed)
    * write-through capability (read on same cycle as write)
    """
    SO=0 # this is actually 2-bit but we ignore 1 bit of it
    CA=1 # CA and CA32
    OV=2 # OV and OV32
    def __init__(self):
        super().__init__(6, 3)
        self.w_ports = {'full_xer': self.full_wr, # 6-bit (masked, 3-en lines)
                        'xer_so': self.write_port("dest1"),
                        'xer_ca': self.write_port("dest2"),
                        'xer_ov': self.write_port("dest3")}
        self.r_ports = {'full_xer': self.full_rd, # 6-bit (masked, 3-en lines)
                        'xer_so': self.read_port("src1"),
                        'xer_ca': self.read_port("src2"),
                        'xer_ov': self.read_port("src3")}


# SPR Regfile
class SPRRegs(Memory, Elaboratable):
    """SPRRegs

    * QTY len(SPRs) 64-bit registers
    * 1R1W
    * binary-indexed but REQUIRES MAPPING
    * write-through capability (read on same cycle as write)
    """
    def __init__(self):
        n_sprs = len(SPR)
        super().__init__(width=64, depth=n_sprs)
        self.w_ports = {'spr1': self.write_port()}
        self.r_ports = {'spr1': self.read_port()}

        # make read/write ports look like RegFileArray
        self.w_ports['spr1'].wen = self.w_ports['spr1'].en
        self.w_ports['spr1'].data_i = self.w_ports['spr1'].data

        self.r_ports['spr1'].ren = self.w_ports['spr1'].en
        self.r_ports['spr1'].data_o = self.w_ports['spr1'].data


# class containing all regfiles: int, cr, xer, fast, spr
class RegFiles:
    def __init__(self):
        self.rf = {}
        for (name, kls) in [('int', IntRegs),
                            ('cr', CRRegs),
                            ('xer', XERRegs),
                            ('fast', FastRegs),
                            ('state', StateRegs),
                            ('spr', SPRRegs),]:
            rf = self.rf[name] = kls()
            setattr(self, name, rf)

    def elaborate_into(self, m, platform):
        for (name, rf) in self.rf.items():
            setattr(m.submodules, name, rf)
        return m

