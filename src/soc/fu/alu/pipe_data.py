from soc.fu.alu.alu_input_record import CompALUOpSubset
from soc.fu.pipe_data import FUBaseData, CommonPipeSpec


class ALUInputData(FUBaseData):
    def __init__(self, pspec):
        super().__init__(pspec, False)
        # convenience
        self.a, self.b = self.ra, self.rb

    @property
    def regspec(self):
        return [('INT', 'ra', self.intrange),  # RA
               ('INT', 'rb', self.intrange),  # RB/immediate
               ('XER', 'xer_so', '32'),  # XER bit 32: SO
               ('XER', 'xer_ca', '34,45')]  # XER bit 34/45: CA/CA32



class ALUOutputData(FUBaseData):
    def __init__(self, pspec):
        super().__init__(pspec, True)
        # convenience
        self.cr0 = self.cr_a

    @property
    def regspec(self):
        return [('INT', 'o', self.intrange),
               ('CR', 'cr_a', '0:3'),
               ('XER', 'xer_ca', '34,45'),  # bit0: ca, bit1: ca32
               ('XER', 'xer_ov', '33,44'),  # bit0: ov, bit1: ov32
               ('XER', 'xer_so', '32')]



class ALUPipeSpec(CommonPipeSpec):
    opsubsetkls = CompALUOpSubset
    regspecklses = (ALUInputData, ALUOutputData)
