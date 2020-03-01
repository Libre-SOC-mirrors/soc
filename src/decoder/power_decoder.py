from nmigen import Module, Elaboratable, Signal
from power_enums import (Function, InternalOp, In1Sel, In2Sel, In3Sel,
                         OutSel, RC, LdstLen, CryIn, get_csv, single_bit_flags,
                         get_signal_name)


class PowerOp:
    def __init__(self):
        self.function_unit = Signal(Function, reset_less=True)
        self.internal_op = Signal(InternalOp, reset_less=True)
        self.in1_sel = Signal(In1Sel, reset_less=True)
        self.in2_sel = Signal(In2Sel, reset_less=True)
        self.in3_sel = Signal(In3Sel, reset_less=True)
        self.out_sel = Signal(OutSel, reset_less=True)
        self.ldst_len = Signal(LdstLen, reset_less=True)
        self.rc_sel = Signal(RC, reset_less=True)
        self.cry_in = Signal(CryIn, reset_less=True)
        for bit in single_bit_flags:
            name = get_signal_name(bit)
            setattr(self, name, Signal(reset_less=True, name=name))

    def _eq(self, row):
        if row is None:
            row = {}
        res = [self.function_unit.eq(Function[row.get('unit', Function.NONE)]),
               self.internal_op.eq(InternalOp[row.get('internal op',
                                                      InternalOp.OP_ILLEGAL)])
               self.in1_sel.eq(In1Sel[row.get('in1', 0)])
               self.in2_sel.eq(In2Sel[row.get('in2', 0)])
               self.in3_sel.eq(In3Sel[row.get('in3', 0)])
               self.out_sel.eq(OutSel[row.get('out', 0)])
               self.ldst_len.eq(LdstLen[row.get('ldst len', 0)])
               self.rc_sel.eq(RC[row.get('rc', 0)])
               self.cry_in.eq(CryIn[row.get('cry in', 0)])
               ]
        for bit in single_bit_flags:
            sig = getattr(self, get_signal_name(bit))
            res.append(sig.eq(0))
        return res

    def ports(self):
        regular = [self.function_unit,
                   self.in1_sel,
                   self.in2_sel,
                   self.in3_sel,
                   self.out_sel,
                   self.ldst_len,
                   self.rc_sel,
                   self.internal_op]
        single_bit_ports = [getattr(self, get_signal_name(x))
                            for x in single_bit_flags]
        return regular + single_bit_ports


class PowerDecoder(Elaboratable):
    def __init__(self, width, csvname):
        self.opcodes = get_csv(csvname)
        self.opcode_in = Signal(width, reset_less=True)

        self.op = PowerOp()

    def elaborate(self, platform):
        m = Module()
        comb = m.d.comb

        with m.Switch(self.opcode_in):
            for row in self.opcodes:
                opcode = int(row['opcode'], 0)
                if not row['unit']:
                    continue
                with m.Case(opcode):
                    comb += self.op._eq(row)
            with m.Default():
                    comb += self.op._eq(None)
        return m

    def ports(self):
        return [self.opcode_in] + self.op.ports()

