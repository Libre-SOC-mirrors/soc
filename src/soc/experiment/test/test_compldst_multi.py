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
from soc.experiment.test.test_compalu_multi import OperandProducer
from soc.fu.ldst.pipe_data import LDSTPipeSpec


class OpSim:
    def __init__(self, dut, sim):
        self.dut = dut
        # create one operand producer for each input port
        self.producers = list()
        for i in range(len(dut.src_i)):
            self.producers.append(OperandProducer(sim, dut, i))

    def issue(self, op, ra=None, rb=None, rc=None,
              zero_a=False, imm=None, update=False,
              byterev=True, signext=False,
              data_len=2, msr_pr=0,
              delays=None):
        assert zero_a == (ra is None), \
            "ra and zero_a are mutually exclusive"
        assert (rb is None) != (imm is None), \
            "rb and imm are mutually exclusive"
        if op == MicrOp.OP_STORE:
            assert rc, "need source operand for store"
        dut = self.dut
        pi = dut.pi
        producers = self.producers
        if ra:
            yield from producers[0].send(ra, delays['ra'])
        if rb:
            yield from producers[1].send(rb, delays['rb'])
        if rc:
            yield from producers[2].send(rc, delays['rc'])
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
        while not (yield pi.addr.ok):
            yield


# FIXME: AttributeError: type object 'LDSTPipeSpec' has no attribute 'regspec'
@unittest.skip('broken')
class TestLDSTCompUnit(unittest.TestCase):

    def test_ldst_compunit(self):
        m = Module()
        pi = PortInterface(name="pi")
        regspec = LDSTPipeSpec.regspec
        dut = LDSTCompUnit(pi, regspec, name="ldst")
        m.submodules.dut = dut
        sim = Simulator(m)
        sim.add_clock(1e-6)
        op = OpSim(dut, sim)
        self.write_gtkw()

        def process():
            yield from op.issue(MicrOp.OP_STORE, ra=1, rb=2, rc=3,
                                delays={'ra': 1, 'rb': 2, 'rc': 5})

        sim.add_sync_process(process)
        sim_writer = sim.write_vcd("test_ldst_compunit.vcd")
        with sim_writer:
            sim.run()

    @classmethod
    def write_gtkw(cls):
        style = {'dec': {'base': 'dec'}}
        traces = [
            'clk',
            ('state latches', [
                'q_opc',
                ('q_src[2:0]', {'bit': 2}),
                ('q_src[2:0]', {'bit': 1}),
                ('q_src[2:0]', {'bit': 0}),
                'q_alu', 'q_adr', 'qn_lod', 'q_sto',
                'q_wri', 'q_upd', 'q_rst',  'q_lsd'
            ]),
            ('operation', [
                ('oper_i_ldst__insn_type', {'display': 'insn_type'}),
                ('oper_i_ldst__ldst_mode', {'display': 'ldst_mode'}),
                ('oper_i_ldst__zero_a', {'display': 'zero_a'}),
                ('oper_i_ldst__imm_data__ok', {'display': 'imm_data_ok'}),
                ('oper_i_ldst__imm_data__data[63:0]', 'dec',
                 {'display': 'imm_data_data'})
            ]),
            'cu_issue_i', 'cu_busy_o',
            ('address ALU', [
                ('cu_rd__rel_o[2:0]', {'bit': 2}),
                ('cu_rd__go_i[2:0]', {'bit': 2}),
                ('src1_i[63:0]', 'dec'),
                ('cu_rd__rel_o[2:0]', {'bit': 1}),
                ('cu_rd__go_i[2:0]', {'bit': 1}),
                ('src2_i[63:0]', 'dec'),
                'alu_valid', 'alu_ok', ('alu_o[63:0]', 'dec'),
                'cu_ad__rel_o', 'cu_ad__go_i',
                'pi_addr_i_ok', ('pi_addr_i[47:0]', 'dec'),
            ]),
            ('store operand', [
                ('cu_rd__rel_o[2:0]', {'bit': 0}),
                ('cu_rd__go_i[2:0]', {'bit': 0}),
                ('src3_i[63:0]', 'dec'),
                'rd_done',
            ]),
            'cu_st__rel_o', 'cu_st__go_i'
        ]
        write_gtkw("test_ldst_compunit.gtkw",
                   "test_ldst_compunit.vcd",
                   traces, style, module="top.dut")


if __name__ == '__main__':
    unittest.main()
