import unittest
from soc.decoder.power_enums import (XER_bits, Function, spr_dict, SPR)

# XXX bad practice: use of global variables
from soc.fu.branch.test.test_pipe_caller import BranchTestCase, get_cu_inputs
from soc.fu.branch.test.test_pipe_caller import test_data

from soc.fu.compunits.compunits import BranchFunctionUnit
from soc.fu.compunits.test.test_compunit import TestRunner

from soc.regfile.util import fast_reg_to_spr # HACK!

"""
    def assert_outputs(self, branch, dec2, sim, prev_nia, code):
"""


class BranchTestRunner(TestRunner):
    def __init__(self, test_data):
        super().__init__(test_data, BranchFunctionUnit, self,
                         Function.BRANCH)

    def get_cu_inputs(self, dec2, sim):
        """naming (res) must conform to BranchFunctionUnit input regspec
        """
        res = yield from get_cu_inputs(dec2, sim)
        return res

    def check_cu_outputs(self, res, dec2, sim, code):
        """naming (res) must conform to BranchFunctionUnit output regspec
        """

        print ("check extra output", repr(code), res)

        # NIA (next instruction address aka PC)
        branch_taken = 'nia' in res
        # TODO - get the old PC, use it to check if the branch was taken
        #     sim_branch_taken = prev_nia != sim.pc.CIA
        #     self.assertEqual(branch_taken, sim_branch_taken, code)
        if branch_taken:
            branch_addr = res['nia']
            self.assertEqual(branch_addr, sim.pc.CIA.value, code)

        # Link SPR
        lk = yield dec2.e.lk
        branch_lk = 'spr2' in res
        self.assertEqual(lk, branch_lk, code)
        if lk:
            branch_lr = res['spr2']
            self.assertEqual(sim.spr['LR'], branch_lr, code)

        # CTR SPR
        ctr_ok = 'spr1' in res
        if ctr_ok:
            ctr = res['spr1']
            self.assertEqual(sim.spr['CTR'], ctr, code)


if __name__ == "__main__":
    unittest.main(exit=False)
    suite = unittest.TestSuite()
    suite.addTest(BranchTestRunner(test_data))

    runner = unittest.TextTestRunner()
    runner.run(suite)
