# Proof of correctness for trap pipeline, main stage


"""
Links:
* https://bugs.libre-soc.org/show_bug.cgi?id=421
* https://libre-soc.org/openpower/isa/fixedtrap/
* https://libre-soc.org/openpower/isa/sprset/
* https://libre-soc.org/openpower/isa/system/
"""


import unittest

from nmigen import Cat, Const, Elaboratable, Module, Signal, signed
from nmigen.asserts import Assert, AnyConst
from nmigen.cli import rtlil

from nmutil.extend import exts
from nmutil.formaltest import FHDLTestCase

from soc.consts import MSR, MSRb, PI, TT, field

from soc.decoder.power_enums import MicrOp

from soc.fu.trap.main_stage import TrapMainStage
from soc.fu.trap.pipe_data import TrapPipeSpec
from soc.fu.trap.trap_input_record import CompTrapOpSubset


class Driver(Elaboratable):
    """
    """

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        rec = CompTrapOpSubset()
        pspec = TrapPipeSpec(id_wid=2)

        m.submodules.dut = dut = TrapMainStage(pspec)

        # frequently used aliases
        op = dut.i.ctx.op
        msr_o, msr_i = dut.o.msr, op.msr
        srr0_o, srr0_i = dut.o.srr0, dut.i.srr0
        srr1_o, srr1_i = dut.o.srr1, dut.i.srr1
        nia_o = dut.o.nia

        comb += op.eq(rec)

        d_fields = dut.fields.FormD
        sc_fields = dut.fields.FormSC

        # start of properties
        with m.Switch(op.insn_type):

            ###############
            # TDI/TWI/TD/TW.  v3.0B p90-91
            ###############
            with m.Case(MicrOp.OP_TRAP):
                to = Signal(len(d_fields.TO))
                comb += to.eq(d_fields.TO[0:-1])

                a_i = Signal(64)
                b_i = Signal(64)
                comb += a_i.eq(dut.i.a)
                comb += b_i.eq(dut.i.b)

                a_s = Signal(signed(64), reset_less=True)
                b_s = Signal(signed(64), reset_less=True)
                a = Signal(64, reset_less=True)
                b = Signal(64, reset_less=True)

                with m.If(op.is_32bit):
                    comb += a_s.eq(exts(a_i, 32, 64))
                    comb += b_s.eq(exts(b_i, 32, 64))
                    comb += a.eq(a_i[0:32])
                    comb += b.eq(b_i[0:32])
                with m.Else():
                    comb += a_s.eq(a_i)
                    comb += b_s.eq(b_i)
                    comb += a.eq(a_i)
                    comb += b.eq(b_i)

                lt_s = Signal(reset_less=True)
                gt_s = Signal(reset_less=True)
                lt_u = Signal(reset_less=True)
                gt_u = Signal(reset_less=True)
                equal = Signal(reset_less=True)

                comb += lt_s.eq(a_s < b_s)
                comb += gt_s.eq(a_s > b_s)
                comb += lt_u.eq(a < b)
                comb += gt_u.eq(a > b)
                comb += equal.eq(a == b)

                trapbits = Signal(5, reset_less=True)
                comb += trapbits.eq(Cat(gt_u, lt_u, equal, gt_s, lt_s))

                take_trap = Signal()
                traptype = op.traptype
                comb += take_trap.eq(traptype.any() | (trapbits & to).any())

                with m.If(take_trap):
                    expected_msr = Signal(len(msr_o.data))
                    comb += expected_msr.eq(op.msr)

                    comb += field(expected_msr, MSRb.IR).eq(0)
                    comb += field(expected_msr, MSRb.DR).eq(0)
                    comb += field(expected_msr, MSRb.FE0).eq(0)
                    comb += field(expected_msr, MSRb.FE1).eq(0)
                    comb += field(expected_msr, MSRb.EE).eq(0)
                    comb += field(expected_msr, MSRb.RI).eq(0)
                    comb += field(expected_msr, MSRb.SF).eq(1)
                    comb += field(expected_msr, MSRb.TM).eq(0)
                    comb += field(expected_msr, MSRb.VEC).eq(0)
                    comb += field(expected_msr, MSRb.VSX).eq(0)
                    comb += field(expected_msr, MSRb.PR).eq(0)
                    comb += field(expected_msr, MSRb.FP).eq(0)
                    comb += field(expected_msr, MSRb.PMM).eq(0)

                    # still wrong.
                    # see https://bugs.libre-soc.org/show_bug.cgi?id=325#c120
                    #
                    # saf2: no it's not.  Proof by substitution:
                    #
                    # field(R,MSRb.TEs,MSRb.TEe).eq(0)
                    # == field(R,53,54).eq(0)
                    # == R[field_slice(53,54)].eq(0)
                    # == R[slice(63-54, (63-53)+1)].eq(0)
                    # == R[slice(9, 11)].eq(0)
                    # == R[9:11].eq(0)
                    #
                    # Also put proof in py-doc for field().

                    comb += field(expected_msr, MSRb.TEs, MSRb.TEe).eq(0)

                    comb += field(expected_msr, MSRb.UND).eq(0)
                    comb += field(expected_msr, MSRb.LE).eq(1)

                    expected_srr1 = Signal(len(srr1_o.data))
                    comb += expected_srr1.eq(op.msr)

                    # Per V3.0B, page 1075
                    comb += field(expected_srr1, 33, 36).eq(0)
                    comb += field(expected_srr1, 42).eq(0)  # TM_BAD_THING
                    comb += field(expected_srr1, 43).eq(traptype[0])    # FP
                    comb += field(expected_srr1, 44).eq(traptype[4])    # ILLEG
                    comb += field(expected_srr1, 45).eq(traptype[1])    # PRIV
                    comb += field(expected_srr1, 46).eq(traptype == 0)  # TRAP
                    comb += field(expected_srr1, 47).eq(traptype[3])    # ADDR

                    comb += [
                        Assert(msr_o.ok),
                        Assert(msr_o.data == expected_msr),
                        Assert(srr0_o.ok),
                        Assert(srr0_o.data == op.cia),
                        Assert(srr1_o.ok),
                        Assert(srr1_o.data == expected_srr1),
                        Assert(nia_o.ok),
                        Assert(nia_o.data == op.trapaddr << 4),
                    ]

            ###################
            # MTMSR
            ###################

            ###################
            # MFMSR
            ###################

            ###################
            # RFID.  v3.0B p955
            ###################
            with m.Case(MicrOp.OP_RFID):
                comb += [
                    Assert(msr_o.ok),
                    Assert(nia_o.ok),
                ]

                # Note: going through the spec pseudo-code, line-by-line,
                # in order, with these assertions.  idea is: compare
                # *directly* against the pseudo-code.  therefore, leave
                # numbering in (from pseudo-code) and add *comments* about
                # which field it is (3 == HV etc.)

                # spec: MSR[51] <- (MSR[3] & SRR1[51]) | ((¬MSR[3] & MSR[51]))
                with m.If(field(msr_i, 3)): # HV
                    comb += Assert(field(msr_o, 51) == field(srr1_i, 51)) # ME
                with m.Else():
                    comb += Assert(field(msr_o, 51) == field(msr_i, 51)) # ME

                # if (MSR[29:31] != 0b010) | (SRR1[29:31] != 0b000) then
                #     MSR[29:31] <- SRR1[29:31]
                with m.If((field(msr_i , 29, 31) != 0b010) |
                          (field(srr1_i, 29, 31) != 0b000)):
                    comb += Assert(field(msr_o.data, 29, 31) ==
                                   field(srr1_i, 29, 31))
                with m.Else():
                    comb += Assert(field(msr_o.data, 29, 31) ==
                                   field(msr_i, 29, 31))

                # check EE (48) IR (58), DR (59): PR (49) will over-ride
                comb += [
                    Assert(
                        field(msr_o, 48) ==
                        field(srr1_i, 48) | field(srr1_i, 49)
                    ),
                    Assert(
                        field(msr_o, 58) ==
                        field(srr1_i, 58) | field(srr1_i, 49)
                    ),
                    Assert(
                        field(msr_o, 59) ==
                        field(srr1_i, 59) | field(srr1_i, 49)
                    ),
                ]

                # remaining bits: straight copy.  don't know what these are:
                # just trust the v3.0B spec is correct.
                comb += [
                    Assert(field(msr_o, 0, 2) == field(srr1_i, 0, 2)),
                    Assert(field(msr_o, 4, 28) == field(srr1_i, 4, 28)),
                    Assert(field(msr_o, 32) == field(srr1_i, 32)),
                    Assert(field(msr_o, 37, 41) == field(srr1_i, 37, 41)),
                    Assert(field(msr_o, 49, 50) == field(srr1_i, 49, 50)),
                    Assert(field(msr_o, 52, 57) == field(srr1_i, 52, 57)),
                    Assert(field(msr_o, 60, 63) == field(srr1_i, 60, 63)),
                ]

                # check NIA against SRR0.  2 LSBs are set to zero (word-align)
                comb += Assert(nia_o.data == Cat(Const(0, 2), dut.i.srr0[2:]))

            #################
            # SC.  v3.0B p952
            #################
            with m.Case(MicrOp.OP_SC):
                expected_msr = Signal(len(msr_o.data))
                comb += expected_msr.eq(op.msr)
                # Unless otherwise documented, these exceptions to the MSR bits
                # are documented in Power ISA V3.0B, page 1063 or 1064.
                # We are not supporting hypervisor or transactional semantics,
                # so we skip enforcing those fields' properties.
                comb += field(expected_msr, MSRb.IR).eq(0)
                comb += field(expected_msr, MSRb.DR).eq(0)
                comb += field(expected_msr, MSRb.FE0).eq(0)
                comb += field(expected_msr, MSRb.FE1).eq(0)
                comb += field(expected_msr, MSRb.EE).eq(0)
                comb += field(expected_msr, MSRb.RI).eq(0)
                comb += field(expected_msr, MSRb.SF).eq(1)
                comb += field(expected_msr, MSRb.TM).eq(0)
                comb += field(expected_msr, MSRb.VEC).eq(0)
                comb += field(expected_msr, MSRb.VSX).eq(0)
                comb += field(expected_msr, MSRb.PR).eq(0)
                comb += field(expected_msr, MSRb.FP).eq(0)
                comb += field(expected_msr, MSRb.PMM).eq(0)
                comb += field(expected_msr, MSRb.TEs, MSRb.TEe).eq(0)
                comb += field(expected_msr, MSRb.UND).eq(0)
                comb += field(expected_msr, MSRb.LE).eq(1)

                comb += [
                    Assert(dut.o.srr0.ok),
                    Assert(srr1_o.ok),
                    Assert(msr_o.ok),

                    Assert(dut.o.srr0.data == (op.cia + 4)[0:64]),
                    Assert(field(srr1_o, 33, 36) == 0),
                    Assert(field(srr1_o, 42, 47) == 0),
                    Assert(field(srr1_o, 0, 32) == field(msr_i, 0, 32)),
                    Assert(field(srr1_o, 37, 41) == field(msr_i, 37, 41)),
                    Assert(field(srr1_o, 48, 63) == field(msr_i, 48, 63)),

                    Assert(msr_o.data == expected_msr),
                ]

        comb += dut.i.ctx.matches(dut.o.ctx)

        return m


class TrapMainStageTestCase(FHDLTestCase):
    def test_formal(self):
        self.assertFormal(Driver(), mode="bmc", depth=10)
        self.assertFormal(Driver(), mode="cover", depth=10)

    def test_ilang(self):
        vl = rtlil.convert(Driver(), ports=[])
        with open("trap_main_stage.il", "w") as f:
            f.write(vl)


if __name__ == '__main__':
    unittest.main()

