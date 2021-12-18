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

from openpower.consts import MSR

from soc.experiment.test import pagetables


class MMUTestCase(TestAccumulatorBase):

    # MMUTEST: initial_msr= 16384
    # msr 16384
    # ISACaller initial_msr 16384
    # FIXME msr does not get passed to LoadStore1
    def case_5_ldst_exception(self):
        lst = [#"mtspr 720,1", # mtspr PRTBL,r1
               "stb 10,0(2)",
               "addi 10,0, 2",
               "lbz 6,0(2)",
              ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1000000
        initial_regs[2] = 0x3456
        initial_regs[3] = 0x4321
        initial_regs[4] = 0x6543
        initial_regs[10] = 0xfe
        initial_mem = {}
        initial_msr = 0 << MSR.PR # must set "problem" state
        initial_msr |= 1 << MSR.DR # set "virtual" state
        initial_sprs = {720: 0x1000000} # PRTBL
        print("MMUTEST: initial_msr=",initial_msr)
        self.add_case(Program(lst, bigendian), initial_regs,
                             initial_mem=initial_mem,
                             initial_sprs=initial_sprs,
                             initial_msr=initial_msr)

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
                              microwatt_mmu=True,
                              rom=pagetables.test1))

    runner = unittest.TextTestRunner()
    runner.run(suite)
