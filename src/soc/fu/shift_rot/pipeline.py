from nmutil.singlepipe import ControlBase
from nmutil.pipemodbase import PipeModBaseChain
from soc.fu.shift_rot.input_stage import ShiftRotInputStage
from soc.fu.shift_rot.main_stage import ShiftRotMainStage
from soc.fu.shift_rot.output_stage import ShiftRotOutputStage

class ShiftRotStart(PipeModBaseChain):
    def get_chain(self):
        inp = ShiftRotInputStage(self.pspec)
        return [inp]

class ShiftRotStage(PipeModBaseChain):
    def get_chain(self):
        main = ShiftRotMainStage(self.pspec)
        return [main]


class ShiftRotStageEnd(PipeModBaseChain):
    def get_chain(self):
        out = ShiftRotOutputStage(self.pspec)
        return [out]


class ShiftRotBasePipe(ControlBase):
    def __init__(self, pspec):
        ControlBase.__init__(self)
        self.pspec = pspec
        self.pipe1 = ShiftRotStart(pspec)
        self.pipe2 = ShiftRotStage(pspec)
        self.pipe3 = ShiftRotStageEnd(pspec)
        self._eqs = self.connect([self.pipe1, self.pipe2, self.pipe3])

    def elaborate(self, platform):
        m = ControlBase.elaborate(self, platform)
        m.submodules.pipe1 = self.pipe1
        m.submodules.pipe2 = self.pipe2
        m.submodules.pipe3 = self.pipe3
        m.d.comb += self._eqs
        return m
