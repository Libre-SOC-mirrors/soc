from soc.fu.shift_rot.sr_input_record import CompSROpSubset
from soc.fu.pipe_data import FUBaseData, CommonPipeSpec
from soc.fu.alu.pipe_data import ALUOutputData


class ShiftRotInputData(FUBaseData):
    def __init__(self, pspec):
        super().__init__(pspec, False)
        # convenience
        self.a, self.b, self.rs = self.ra, self.rb, self.rc

    @property
    def regspec(self):
        return [('INT', 'ra', self.intrange),  # RA
               ('INT', 'rb', self.intrange),  # RB/immediate
               ('INT', 'rc', self.intrange),  # RB/immediate
               ('XER', 'xer_so', '32'), # XER bit 32: SO
               ('XER', 'xer_ca', '34,45')] # XER bit 34/45: CA/CA32


# input to shiftrot final stage (common output)
class ShiftRotOutputData(FUBaseData):
    def __init__(self, pspec):
        super().__init__(pspec, True)
        # convenience
        self.cr0 = self.cr_a

    @property
    def regspec(self):
        return [('INT', 'o', self.intrange),
               ('CR', 'cr_a', '0:3'),
               ('XER', 'xer_so', '32'),    # bit0: so
               ('XER', 'xer_ca', '34,45'), # XER bit 34/45: CA/CA32
               ]


# output from shiftrot final stage (common output) - note that XER.so
# is *not* included (the only reason it's in the input is because of CR0)
class ShiftRotOutputDataFinal(FUBaseData):
    def __init__(self, pspec):
        super().__init__(pspec, True)
        # convenience
        self.cr0 = self.cr_a

    @property
    def regspec(self):
        return [('INT', 'o', self.intrange),
               ('CR', 'cr_a', '0:3'),
               ('XER', 'xer_ca', '34,45'), # XER bit 34/45: CA/CA32
               ]


class ShiftRotPipeSpec(CommonPipeSpec):
    regspecklses = (ShiftRotInputData, ShiftRotOutputDataFinal)
    opsubsetkls = CompSROpSubset
