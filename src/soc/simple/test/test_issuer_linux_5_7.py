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
from soc.experiment.test import pagetables


from openpower.simulator.program import Program
from openpower.endian import bigendian
from openpower.test.common import TestAccumulatorBase

from openpower.consts import MSR

from soc.experiment.test import pagetables


class MMUTestCase(TestAccumulatorBase):

    def cse_first_vm_enabled(self):
        lst = [
               "std 6,0(2)",
              ]

        # set up regs
        initial_regs = [0] * 32
        initial_regs[2] = 0xc0000000005fc190
        initial_regs[6] = 0x0101

        # memory same as microwatt test
        initial_mem = pagetables.microwatt_linux_5_7_boot

        # set virtual and non-privileged
        # msr: 8000000000000011
        initial_msr = 0 << MSR.PR # must set "problem" state
        initial_msr |= 1 << MSR.LE # little-endian
        initial_msr |= 1 << MSR.SF # 64-bit
        initial_msr |= 1 << MSR.DR # set "virtual" state for data

        # set PRTBL to 0xe000000
        initial_sprs = {720: 0xe000000, # PRTBL
                        48: 1       # PIDR
                        } 

        print("MMUTEST: initial_msr=",initial_msr)
        self.add_case(Program(lst, bigendian), initial_regs,
                             initial_mem=initial_mem,
                             initial_sprs=initial_sprs,
                             initial_msr=initial_msr)


    def case_first_vm_enabled_2(self):
        lst = [
               "std 6,0(2)",
              ]

        # set up regs
        initial_regs = [0] * 32
        initial_regs[2] = 0xc000000000598000
        initial_regs[6] = 0x0101

        # memory same as microwatt test
        initial_mem = pagetables.microwatt_linux_5_7_boot

        # set virtual and non-privileged
        # msr: 8000000000000011
        initial_msr = 0 << MSR.PR # must set "problem" state
        initial_msr |= 1 << MSR.LE # little-endian
        initial_msr |= 1 << MSR.SF # 64-bit
        initial_msr |= 1 << MSR.DR # set "virtual" state for data

        # set PRTBL to 0xe000000
        initial_sprs = {720: 0xe00000c, # PRTBL
                        48: 1       # PIDR
                        }

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
                              rom=pagetables.microwatt_linux_5_7_boot))

    runner = unittest.TextTestRunner()
    runner.run(suite)
