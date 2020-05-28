# This is the proof for Regfile class from regfile/regfile.py

from nmigen import (Module, Signal, Elaboratable, Mux, Cat, Repl,
                    signed, ResetSignal)
from nmigen.asserts import (Assert, AnyConst, Assume, Cover, Initial,
                            Rose, Fell, Stable, Past)
from nmigen.test.utils import FHDLTestCase
from nmigen.cli import rtlil
import unittest

from soc.regfile.regfile import Register


class Driver(Register):
    def __init__(self, writethru=True):
        super().__init__(8, writethru)

    def elaborate(self, platform):
        m = super().elaborate(platform)
        comb = m.d.comb
        sync = m.d.sync

        width     = self.width
        writethru = self.writethru
        _rdports  = self._rdports
        _wrports  = self._wrports
        reg       = self.reg

        for i in range(1): # just do one for now
            self.read_port(f"{i}")
            self.write_port(f"{i}")

        comb += _wrports[0].data_i.eq(AnyConst(8))
        comb += _wrports[0].wen.eq(AnyConst(1))
        comb += _rdports[0].ren.eq(AnyConst(1))
        sync += reg.eq(AnyConst(8))

        rst = ResetSignal()

        init = Initial()

        # Most likely incorrect 4-way truth table
        #
        # rp.ren  wp.wen  rp.data_o            reg
        # 0       0       zero                 should be previous value
        # 0       1       zero                 wp.data_i
        # 1       0       reg                  should be previous value
        # 1       1       wp.data_i            wp.data_i

        with m.If(init):
            comb += Assume(rst == 1)

        with m.Else():
            comb += Assume(rst == 0)
            if writethru:
                for i in range(len(_rdports)):
                    with m.If(_rdports[i].ren):
                        with m.If(_wrports[i].wen):
                            pass
                            #comb += Assert(_rdports[i].data_o == _wrports[i].data_i)
                        with m.Else():
                            pass
                            #comb += Assert(_rdports[i].data_o == 0)
                    with m.Else():
                        #comb += Assert(_rdports[i].data_o == reg)
                        pass

                for i in range(len(_wrports)):
                    with m.If(Past(_wrports[i].wen)):
                        #comb += Assert(reg == Past(_wrports[i].data_i))
                        pass
                    with m.Else():
                        # if wen not set, reg should not change
                        comb += Assert(reg == Past(reg))
            else:
                pass

        return m


class TestCase(FHDLTestCase):
    def test_formal(self):
        module = Driver()
        self.assertFormal(module, mode="bmc", depth=2)
        self.assertFormal(module, mode="cover", depth=2)

    def test_ilang(self):
        dut = Driver()
        vl = rtlil.convert(dut, ports=[])
        with open("regfile.il", "w") as f:
            f.write(vl)


if __name__ == '__main__':
    unittest.main()
