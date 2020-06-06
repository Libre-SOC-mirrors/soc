from nmigen import Module, Signal
from nmigen.back.pysim import Simulator, Delay, Settle
from nmutil.formaltest import FHDLTestCase
from nmigen.cli import rtlil
import unittest
from soc.decoder.isa.caller import ISACaller, special_sprs
from soc.decoder.power_decoder import (create_pdecode)
from soc.decoder.power_decoder2 import (PowerDecode2)
from soc.decoder.power_enums import (XER_bits, Function, InternalOp, CryIn)
from soc.decoder.selectable_int import SelectableInt
from soc.simulator.program import Program
from soc.decoder.isa.all import ISA


from soc.fu.test.common import TestCase
from soc.fu.ldst.pipe_data import LDSTPipeSpec
import random


def get_cu_inputs(dec2, sim):
    """naming (res) must conform to LDSTFunctionUnit input regspec
    """
    res = {}

    # RA
    reg1_ok = yield dec2.e.read_reg1.ok
    if reg1_ok:
        data1 = yield dec2.e.read_reg1.data
        res['ra'] = sim.gpr(data1).value

    # RB (or immediate)
    reg2_ok = yield dec2.e.read_reg2.ok
    if reg2_ok:
        data2 = yield dec2.e.read_reg2.data
        res['rb'] = sim.gpr(data2).value

    # RC
    reg3_ok = yield dec2.e.read_reg3.ok
    if reg3_ok:
        data3 = yield dec2.e.read_reg3.data
        res['rc'] = sim.gpr(data3).value

    # XER.so
    oe = yield dec2.e.oe.data[0] & dec2.e.oe.ok
    if oe:
        so = 1 if sim.spr['XER'][XER_bits['SO']] else 0
        res['xer_so'] = so

    return res


class LDSTTestCase(FHDLTestCase):
    test_data = []

    def __init__(self, name):
        super().__init__(name)
        self.test_name = name

    def run_tst_program(self, prog, initial_regs=None, initial_sprs=None):
        tc = TestCase(prog, self.test_name, initial_regs, initial_sprs)
        self.test_data.append(tc)

    def test_load_store(self):
        lst = ["stw 2, 0(1)",
               "lwz 3, 0(1)"]
        initial_regs = [0] * 32
        initial_regs[1] = 0x0004
        initial_regs[2] = 0x0008
        self.run_tst_program(Program(lst), initial_regs)

    def test_ilang(self):
        pspec = LDSTPipeSpec(id_wid=2)
        alu = LDSTBasePipe(pspec)
        vl = rtlil.convert(alu, ports=alu.ports())
        with open("ldst_pipeline.il", "w") as f:
            f.write(vl)


