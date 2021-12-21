from nmigen import Module, Signal
from soc.simple.test.test_issuer import TestRunner
from openpower.simulator.program import Program
from openpower.endian import bigendian
import unittest

from openpower.test.common import (
    TestAccumulatorBase, skip_case, TestCase, ALUHelpers)

# this test case takes about half a minute to run on my Talos II
class MMUTestCase(TestAccumulatorBase):
    # MMU on microwatt handles MTSPR, MFSPR, DCBZ and TLBIE.
    # libre-soc has own SPR unit
    # other instructions here -> must be load/store

    def case_mmu_dar(self):
        lst = [
                 "mfspr 1, 720",     # DAR to reg 1
                "addi 7, 0, 1",
                "mtspr 19, 3",      # reg 3 to DAR
                "mulli 7, 0, 1",
              ]

        initial_regs = [0] * 32
        initial_regs[1] = 0x2
        initial_regs[3] = 0x5

        initial_sprs = {'DAR': 0x87654321,
                        }
        self.add_case(Program(lst, bigendian),
                      initial_regs, initial_sprs)

    def cse_mmu_ldst(self):
        lst = [
                "dcbz 1,2",
                "tlbie 0,0,0,0,0", # RB,RS,RIC,PRS,R
                "mtspr 18, 1",     # reg 1 to DSISR
                "mtspr 19, 2",     # reg 2 to DAR
                "mfspr 5, 18",     # DSISR to reg 5
                "mfspr 6, 19",     # DAR to reg 6
                "mtspr 48, 3",    # set MMU PID
                "mtspr 720, 4",    # set MMU PRTBL
                "lhz 3, 0(1)",     # load some data
                "addi 7, 0, 1"
              ]

        initial_regs = [0] * 32
        initial_regs[1] = 0x2
        initial_regs[2] = 0x2020
        initial_regs[3] = 5
        initial_regs[4] = 0xDEADBEEF

        initial_sprs = {'DSISR': 0x12345678, 'DAR': 0x87654321,
                        'PIDR': 0xabcd, 'PRTBL': 0x0def}
        self.add_case(Program(lst, bigendian),
                      initial_regs, initial_sprs)


if __name__ == "__main__":
    mem = {}
    unittest.main(exit=False)
    suite = unittest.TestSuite()
    suite.addTest(TestRunner(MMUTestCase().test_data,
                             microwatt_mmu=True,
                             svp64=False,
                             rom=mem))
    runner = unittest.TextTestRunner()
    runner.run(suite)
