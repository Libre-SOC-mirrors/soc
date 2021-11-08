"""simple core test, runs instructions from a TestMemory

related bugs:

 * https://bugs.libre-soc.org/show_bug.cgi?id=363
"""

# NOTE: to use cxxsim, export NMIGEN_SIM_MODE=cxxsim from the shell
# Also, check out the cxxsim nmigen branch, and latest yosys from git

import unittest
import sys

# here is the logic which takes test cases and "executes" them.
# in this instance (TestRunner) its job is to instantiate both
# a Libre-SOC nmigen-based HDL instance and an ISACaller python
# simulator.  it's also responsible for performing the single
# step and comparison.
from soc.simple.test.test_runner import TestRunner

#@platen:bookmarks
#src/openpower/test/runner.py:class TestRunnerBase(FHDLTestCase):

# test with MMU
from openpower.test.mmu.mmu_cases import MMUTestCase
from openpower.test.mmu.mmu_rom_cases import MMUTestCaseROM, default_mem
from openpower.test.ldst.ldst_cases import LDSTTestCase
from openpower.test.ldst.ldst_exc_cases import LDSTExceptionTestCase
#from openpower.simulator.test_sim import (GeneralTestCases, AttnTestCase)

from openpower.simulator.program import Program
from openpower.endian import bigendian
from openpower.test.common import TestAccumulatorBase

class MMUTestCase(TestAccumulatorBase):

    # now working correctly
    def case_1_dcbz(self):
        lst = ["dcbz 1, 2",  # MMUTEST.DCBZ: EA from adder 12
               "dcbz 1, 3"]  # MMUTEST.DCBZ: EA from adder 11
        initial_regs = [0] * 32
        initial_regs[1] = 0x0004
        initial_regs[2] = 0x0008
        initial_regs[3] = 0x0007
        initial_mem = {}
        self.add_case(Program(lst, bigendian), initial_regs,
                             initial_mem=initial_mem)

    # MMUTEST: OP_TLBIE: insn_bits=39
    def case_2_tlbie(self):
        lst = ["tlbie 1,1,1,1,1"] # tlbie   RB,RS,RIC,PRS,R
        initial_regs = [0] * 32
        initial_mem = {}
        self.add_case(Program(lst, bigendian), initial_regs,
                             initial_mem=initial_mem)

    # OP_MTSPR: spr=720
    def case_3_mtspr(self):
        lst = ["mtspr 720,1"] # mtspr PRTBL,r1
        initial_regs = [0] * 32
        initial_regs[1] = 0x1234
        initial_mem = {}
        self.add_case(Program(lst, bigendian), initial_regs,
                             initial_mem=initial_mem)

    # OP_MFSPR: spr=18/19
    def case_4_mfspr(self):
        lst = ["mfspr 1,18", # mtspr r1,DSISR
               "mfspr 2,19"] # mtspr r2,DAR
        initial_regs = [0] * 32
        initial_regs[1] = 0x1234
        initial_regs[2] = 0x3456
        initial_mem = {}
        self.add_case(Program(lst, bigendian), initial_regs,
                             initial_mem=initial_mem)

if __name__ == "__main__":
    svp64 = True
    if len(sys.argv) == 2:
        if sys.argv[1] == 'nosvp64':
            svp64 = False
        sys.argv.pop()

    print ("SVP64 test mode enabled", svp64)

    unittest.main(exit=False)
    suite = unittest.TestSuite()

    # MMU/DCache integration tests
    suite.addTest(TestRunner(MMUTestCase().test_data, svp64=svp64,
                              microwatt_mmu=True))

    runner = unittest.TextTestRunner()
    runner.run(suite)
