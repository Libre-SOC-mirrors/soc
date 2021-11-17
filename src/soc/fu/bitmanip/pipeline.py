from nmutil.singlepipe import ControlBase
from nmutil.pipemodbase import PipeModBaseChain
from soc.fu.bitmanip.input_stage import BitManipInputStage
from soc.fu.bitmanip.main_stage import BitManipMainStage
from soc.fu.bitmanip.output_stage import BitManipOutputStage


class BitManipStages(PipeModBaseChain):
    def get_chain(self):
        inp = BitManipInputStage(self.pspec)
        main = BitManipMainStage(self.pspec)
        return [inp, main]


class BitManipStageEnd(PipeModBaseChain):
    def get_chain(self):
        out = BitManipOutputStage(self.pspec)
        return [out]


class BitManipBasePipe(ControlBase):
    def __init__(self, pspec):
        ControlBase.__init__(self)
        self.pspec = pspec
        self.pipe1 = BitManipStages(pspec)
        self.pipe2 = BitManipStageEnd(pspec)
        self._eqs = self.connect([self.pipe1, self.pipe2])

    def elaborate(self, platform):
        m = ControlBase.elaborate(self, platform)
        m.submodules.pipe1 = self.pipe1
        m.submodules.pipe2 = self.pipe2
        m.d.comb += self._eqs
        return m
