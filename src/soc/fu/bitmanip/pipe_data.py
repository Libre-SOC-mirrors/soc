from soc.fu.bitmanip.input_record import CompBitManipOpSubset
from soc.fu.pipe_data import FUBaseData, CommonPipeSpec
from soc.fu.alu.pipe_data import ALUOutputData


class BitManipInputData(FUBaseData):
    regspec = [
        ('INT', 'ra', '0:63'),      # RA
        ('INT', 'rb', '0:63'),      # RB
        ('INT', 'rc', '0:63'),      # RC
        ('XER', 'xer_so', '32'),  # XER bit 32: SO
    ]

    def __init__(self, pspec):
        super().__init__(pspec, False)


# input to bitmanip final stage (common output)
class BitManipOutputData(FUBaseData):
    regspec = [
        ('INT', 'o', '0:63'),        # RT
        ('CR', 'cr_a', '0:3'),
        ('XER', 'xer_so', '32'),    # bit0: so
    ]

    def __init__(self, pspec):
        super().__init__(pspec, True)
        # convenience
        self.cr0 = self.cr_a


# output from bitmanip final stage (common output) - note that XER.so
# is *not* included (the only reason it's in the input is because of CR0)
class BitManipOutputDataFinal(FUBaseData):
    regspec = [('INT', 'o', '0:63'),        # RT
               ('CR', 'cr_a', '0:3'),
               ]

    def __init__(self, pspec):
        super().__init__(pspec, True)
        # convenience
        self.cr0 = self.cr_a


class BitManipPipeSpec(CommonPipeSpec):
    regspec = (BitManipInputData.regspec, BitManipOutputDataFinal.regspec)
    opsubsetkls = CompBitManipOpSubset
