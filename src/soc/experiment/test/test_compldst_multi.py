"""Self-contained unit test for the Load/Store CompUnit
"""

import unittest
from nmigen import Module
from nmigen.sim import Simulator
from nmutil.gtkw import write_gtkw

from openpower.consts import MSR
from openpower.decoder.power_enums import MicrOp, LDSTMode

from soc.experiment.compldst_multi import LDSTCompUnit
from soc.experiment.pimem import PortInterface
from soc.fu.ldst.pipe_data import LDSTPipeSpec


class OpSim:
    def __init__(self, dut):
        self.dut = dut

    def issue(self, op, zero_a=False, imm=None, update=False,
              byterev=True, signext=False,
              data_len=2, msr_pr=0):
        dut = self.dut
        yield dut.oper_i.insn_type.eq(op)
        yield dut.oper_i.data_len.eq(data_len)
        yield dut.oper_i.zero_a.eq(zero_a)
        yield dut.oper_i.byte_reverse.eq(byterev)
        yield dut.oper_i.sign_extend.eq(signext)
        if imm is not None:
            yield dut.oper_i.imm_data.data.eq(imm)
            yield dut.oper_i.imm_data.ok.eq(1)
        if update:
            yield dut.oper_i.ldst_mode.eq(LDSTMode.update)
        yield dut.oper_i.msr[MSR.PR].eq(msr_pr)
        yield dut.issue_i.eq(1)
        yield
        yield dut.issue_i.eq(0)
        # deactivate decoder inputs along with issue_i, so we can be sure they
        # were latched at the correct cycle
        yield dut.oper_i.insn_type.eq(0)
        yield dut.oper_i.data_len.eq(0)
        yield dut.oper_i.zero_a.eq(0)
        yield dut.oper_i.byte_reverse.eq(0)
        yield dut.oper_i.sign_extend.eq(0)
        yield dut.oper_i.imm_data.data.eq(0)
        yield dut.oper_i.imm_data.ok.eq(0)
        yield dut.oper_i.ldst_mode.eq(LDSTMode.NONE)
        yield dut.oper_i.msr[MSR.PR].eq(0)


class TestLDSTCompUnit(unittest.TestCase):

    def test_ldst_compunit(self):
        m = Module()
        pi = PortInterface(name="pi")
        regspec = LDSTPipeSpec.regspec
        dut = LDSTCompUnit(pi, regspec, name="ldst")
        m.submodules.dut = dut
        sim = Simulator(m)
        sim.add_clock(1e-6)
        op = OpSim(dut)
        self.write_gtkw()

        def process():
            yield from op.issue(MicrOp.OP_STORE)

        sim.add_sync_process(process)
        sim_writer = sim.write_vcd("test_ldst_compunit.vcd")
        with sim_writer:
            sim.run()

    @classmethod
    def write_gtkw(cls):
        traces = [
            'clk',
            ('operation', [
                ('oper_i_ldst__insn_type', {'display': 'insn_type'}),
                ('oper_i_ldst__ldst_mode', {'display': 'ldst_mode'}),
                ('oper_i_ldst__zero_a', {'display': 'zero_a'}),
                ('oper_i_ldst__imm_data__ok', {'display': 'imm_data_ok'}),
                ('oper_i_ldst__imm_data__data[63:0]',
                 {'display': 'imm_data_data', 'base': 'dec'})
            ]),
            ('cu_issue_i', {'display': 'issue_i'}),
            ('cu_busy_o', {'display': 'busy_o'})
        ]
        write_gtkw("test_ldst_compunit.gtkw",
                   "test_ldst_compunit.vcd",
                   traces, module="top.dut")


if __name__ == '__main__':
    unittest.main()
