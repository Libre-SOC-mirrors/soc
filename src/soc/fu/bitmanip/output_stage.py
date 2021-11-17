# This stage is intended to handle the gating of carry and overflow
# out, summary overflow generation, and updating the condition
# register
from soc.fu.common_output_stage import CommonOutputStage
from soc.fu.bitmanip.pipe_data import (BitManipOutputData,
                                       BitManipOutputDataFinal)


class BitManipOutputStage(CommonOutputStage):

    def ispec(self):
        return BitManipOutputData(self.pspec)

    def ospec(self):
        return BitManipOutputDataFinal(self.pspec)
