"""dcbz test case

related bugs:

 * https://bugs.libre-soc.org/show_bug.cgi?id=51
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

##########
from openpower.simulator.program import Program
from openpower.endian import bigendian
from openpower.test.common import TestAccumulatorBase

class DCBZTestCase(TestAccumulatorBase):

    def case_1_dcbz(self):
        lst = ["dcbz 1, 2"]
        initial_regs = [0] * 32
        initial_regs[1] = 0x0004
        initial_regs[2] = 0x0008
        initial_mem = {0x0000: (0x5432123412345678, 8),
                       0x0008: (0xabcdef0187654321, 8),
                       0x0020: (0x1828384822324252, 8),
                        }
        self.add_case(Program(lst, bigendian), initial_regs,
                             initial_mem=initial_mem)
##########


if __name__ == "__main__":
    svp64 = False

    unittest.main(exit=False)
    suite = unittest.TestSuite()

    # add other test cases later
    suite.addTest(TestRunner(DCBZTestCase().test_data, svp64=svp64,
                              microwatt_mmu=True))

    runner = unittest.TextTestRunner()
    runner.run(suite)
