from nmutil.singlepipe import ControlBase
from nmutil.pipemodbase import PipeModBaseChain
from soc.fu.alu.input_stage import ALUInputStage
from soc.fu.alu.main_stage import ALUMainStage
from soc.fu.alu.output_stage import ALUOutputStage


class ALUStages(PipeModBaseChain):
    def get_chain(self):
        inp = ALUInputStage(self.pspec)
        main = ALUMainStage(self.pspec)
        out = ALUOutputStage(self.pspec)
        return [inp, main, out]


class ALUBasePipe(ControlBase):
    def __init__(self, pspec):
        ControlBase.__init__(self)
        self.pspec = pspec
        self.pipe1 = ALUStages(pspec)
        self._eqs = self.connect([self.pipe1])

    def elaborate(self, platform):
        m = ControlBase.elaborate(self, platform)
        m.submodules.pipe1 = self.pipe1
        m.d.comb += self._eqs
        return m

class ALUStages1(PipeModBaseChain):
    def get_chain(self):
        inp = ALUInputStage(self.pspec)
        return [inp]

class ALUStages2(PipeModBaseChain):
    def get_chain(self):
        main = ALUMainStage(self.pspec)
        return [main]


class ALUStages3(PipeModBaseChain):
    def get_chain(self):
        out = ALUOutputStage(self.pspec)
        return [out]


class ALUBasePipe(ControlBase):
    def __init__(self, pspec):
        ControlBase.__init__(self)
        self.pspec = pspec
        self.pipe1 = ALUStages1(pspec)
        self.pipe2 = ALUStages2(pspec)
        self.pipe3 = ALUStages3(pspec)
        self._eqs = self.connect([self.pipe1, self.pipe2, self.pipe3])

    def elaborate(self, platform):
        m = ControlBase.elaborate(self, platform)
        m.submodules.logical_pipe1 = self.pipe1
        m.submodules.logical_pipe2 = self.pipe2
        m.submodules.logical_pipe3 = self.pipe3
        m.d.comb += self._eqs
        return m

