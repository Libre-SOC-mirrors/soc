from soc.fu.pipe_data import FUBaseData
from soc.fu.alu.pipe_data import ALUOutputData, CommonPipeSpec
from soc.fu.logical.logical_input_record import CompLogicalOpSubset


# input (and output) for logical initial stage (common input)
class LogicalInputData(FUBaseData):
    def __init__(self, pspec):
        super().__init__(pspec, False)
        # convenience
        self.a, self.b = self.ra, self.rb

    @property
    def regspec(self):
        return [('INT', 'ra', self.intrange),  # RA
               ('INT', 'rb', self.intrange),  # RB/immediate
               ('XER', 'xer_so', '32'),    # bit0: so
               ]

# input to logical final stage (common output)
class LogicalOutputData(FUBaseData):
    def __init__(self, pspec):
        super().__init__(pspec, True)
        # convenience
        self.cr0 = self.cr_a

    @property
    def regspec(self):
        return [('INT', 'o', self.intrange),
               ('CR', 'cr_a', '0:3'),
               ('XER', 'xer_so', '32'),    # bit0: so
               ]


# output from logical final stage (common output) - note that XER.so
# is *not* included (the only reason it's in the input is because of CR0)
class LogicalOutputDataFinal(FUBaseData):
    def __init__(self, pspec):
        super().__init__(pspec, True)
        # convenience
        self.cr0 = self.cr_a
    @property
    def regspec(self):
        return [('INT', 'o', self.intrange),
               ('CR', 'cr_a', '0:3'),
               ]


class LogicalPipeSpec(CommonPipeSpec):
    regspecklses = (LogicalInputData, LogicalOutputDataFinal)
    opsubsetkls = CompLogicalOpSubset
