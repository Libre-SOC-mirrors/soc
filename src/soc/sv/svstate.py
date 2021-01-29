# SPDX-License-Identifier: LGPLv3+
# Copyright (C) 2021 Luke Kenneth Casson Leighton <lkcl@lkcl.net>
# Funded by NLnet http://nlnet.nl
"""SVSATE SPR Record.  actually a peer of PC (CIA/NIA) and MSR

https://libre-soc.org/openpower/sv/sprs/

| Field | Name     | Description           |
| ----- | -------- | --------------------- |
| 0:6   | maxvl    | Max Vector Length     |
| 7:13  |    vl    | Vector Length         |
| 14:20 | srcstep  | for srcstep = 0..VL-1 |
| 21:27 | dststep  | for dststep = 0..VL-1 |
| 28:29 | subvl    | Sub-vector length     |
| 30:31 | svstep   | for svstep = 0..SUBVL-1  |
"""

from nmigen import Record

class SVSTATERec(Record):
    def __init__(self, name=None):
        Record.__init__(self, layout=[("maxvl"     : 7),
                                      ("vl"        : 7),
                                      ("srcstep"   : 7),
                                      ("dststep"   : 7),
                                      ("subvl"     : 2),
                                      ("svstep"    : 2)], name=name)
    def ports(self):
        return [self.maxvl, self.vl, self.srcstep, self.dststep, self.subvl,
                self.svstep]

