from nmutil.singlepipe import ControlBase
from nmutil.pipemodbase import PipeModBaseChain
from soc.fu.branch.main_stage import BranchMainStage
from nmutil.pipemodbase import PipeModBase
from soc.fu.branch.pipe_data import BranchInputData
from nmigen import Module

# gives a 1-clock delay to stop combinatorial link between in and out
class DummyBranchStage(PipeModBase):
    def __init__(self, pspec): super().__init__(pspec, "dummy")
    def ispec(self): return BranchInputData(self.pspec)
    def ospec(self): return BranchInputData(self.pspec)

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(self.i) # pass-through output
        return m

class BranchDummyStages(PipeModBaseChain):
    def get_chain(self):
        dummy = DummyBranchStage(self.pspec)
        return [dummy]


class BranchStages(PipeModBaseChain):
    def get_chain(self):
        main = BranchMainStage(self.pspec)
        return [main]


class BranchBasePipe(ControlBase):
    def __init__(self, pspec):
        ControlBase.__init__(self)
        self.pspec = pspec
        self.pipe1 = BranchDummyStages(pspec)
        self.pipe2 = BranchStages(pspec)
        self._eqs = self.connect([self.pipe1, self.pipe2])

    def elaborate(self, platform):
        m = ControlBase.elaborate(self, platform)
        m.submodules.pipe1 = self.pipe1
        m.submodules.pipe2 = self.pipe2
        m.d.comb += self._eqs
        return m
