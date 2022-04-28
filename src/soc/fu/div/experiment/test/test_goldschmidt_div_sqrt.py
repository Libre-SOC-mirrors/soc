# SPDX-License-Identifier: LGPL-3-or-later
# Copyright 2022 Jacob Lifshay programmerjake@gmail.com

# Funded by NLnet Assure Programme 2021-02-052, https://nlnet.nl/assure part
# of Horizon 2020 EU Programme 957073.

import math
import unittest
from nmutil.formaltest import FHDLTestCase
from nmutil.sim_util import do_sim
from nmigen.sim import Tick, Delay
from nmigen.hdl.ast import Signal
from nmigen.hdl.dsl import Module
from soc.fu.div.experiment.goldschmidt_div_sqrt import (
    GoldschmidtDivHDL, GoldschmidtDivParams, ParamsNotAccurateEnough,
    goldschmidt_div, FixedPoint, RoundDir, goldschmidt_sqrt_rsqrt)


class TestFixedPoint(FHDLTestCase):
    def test_str_roundtrip(self):
        for frac_wid in range(8):
            for bits in range(-1 << 9, 1 << 9):
                with self.subTest(bits=hex(bits), frac_wid=frac_wid):
                    value = FixedPoint(bits, frac_wid)
                    round_trip_value = FixedPoint.cast(str(value))
                    self.assertEqual(value, round_trip_value)

    @staticmethod
    def trap(f):
        try:
            return f(), None
        except (ValueError, ZeroDivisionError) as e:
            return None, e.__class__.__name__

    def test_sqrt(self):
        for frac_wid in range(8):
            for bits in range(1 << 9):
                for round_dir in RoundDir:
                    radicand = FixedPoint(bits, frac_wid)
                    expected_f = math.sqrt(float(radicand))
                    expected = self.trap(lambda: FixedPoint.with_frac_wid(
                        expected_f, frac_wid, round_dir))
                    with self.subTest(radicand=repr(radicand),
                                      round_dir=str(round_dir),
                                      expected=repr(expected)):
                        result = self.trap(lambda: radicand.sqrt(round_dir))
                        self.assertEqual(result, expected)

    def test_rsqrt(self):
        for frac_wid in range(8):
            for bits in range(1, 1 << 9):
                for round_dir in RoundDir:
                    radicand = FixedPoint(bits, frac_wid)
                    expected_f = 1 / math.sqrt(float(radicand))
                    expected = self.trap(lambda: FixedPoint.with_frac_wid(
                        expected_f, frac_wid, round_dir))
                    with self.subTest(radicand=repr(radicand),
                                      round_dir=str(round_dir),
                                      expected=repr(expected)):
                        result = self.trap(lambda: radicand.rsqrt(round_dir))
                        self.assertEqual(result, expected)


class TestGoldschmidtDiv(FHDLTestCase):
    def test_case1(self):
        with self.assertRaises(ParamsNotAccurateEnough):
            GoldschmidtDivParams(io_width=3, extra_precision=2,
                                 table_addr_bits=3, table_data_bits=5,
                                 iter_count=2)

    def test_case2(self):
        with self.assertRaises(ParamsNotAccurateEnough):
            GoldschmidtDivParams(io_width=4, extra_precision=1,
                                 table_addr_bits=1, table_data_bits=5,
                                 iter_count=1)

    def tst(self, io_width):
        assert isinstance(io_width, int)
        params = GoldschmidtDivParams.get(io_width)
        with self.subTest(params=str(params)):
            for d in range(1, 1 << io_width):
                for n in range(d << io_width):
                    expected_q, expected_r = divmod(n, d)
                    with self.subTest(n=hex(n), d=hex(d),
                                      expected_q=hex(expected_q),
                                      expected_r=hex(expected_r)):
                        q, r = goldschmidt_div(n, d, params)
                        with self.subTest(q=hex(q), r=hex(r)):
                            self.assertEqual((q, r), (expected_q, expected_r))

    @unittest.skip("hdl/simulation currently broken")
    def tst_sim(self, io_width, cases=None, pipe_reg_indexes=(),
                sync_rom=False):
        # FIXME: finish getting hdl/simulation to work
        assert isinstance(io_width, int)
        params = GoldschmidtDivParams.get(io_width)
        m = Module()
        dut = GoldschmidtDivHDL(params, pipe_reg_indexes=pipe_reg_indexes,
                                sync_rom=sync_rom)
        m.submodules.dut = dut
        # make sync domain get added
        m.d.sync += Signal().eq(0)

        def iter_cases():
            if cases is not None:
                yield from cases
                return
            for d in range(1, 1 << io_width):
                for n in range(d << io_width):
                    yield (n, d)

        def inputs_proc():
            yield Tick()
            for n, d in iter_cases():
                yield dut.n.eq(n)
                yield dut.d.eq(d)
                yield Tick()

        def check_outputs():
            yield Tick()
            for _ in range(dut.total_pipeline_registers):
                yield Tick()
            for n, d in iter_cases():
                yield Delay(0.1e-6)
                expected_q, expected_r = divmod(n, d)
                with self.subTest(n=hex(n), d=hex(d),
                                  expected_q=hex(expected_q),
                                  expected_r=hex(expected_r)):
                    q = yield dut.q
                    r = yield dut.r
                    with self.subTest(q=hex(q), r=hex(r)):
                        self.assertEqual((q, r), (expected_q, expected_r))
                yield Tick()

        with self.subTest(params=str(params)):
            with do_sim(self, m, (dut.n, dut.d, dut.q, dut.r)) as sim:
                sim.add_clock(1e-6)
                sim.add_process(inputs_proc)
                sim.add_process(check_outputs)
                sim.run()

    def test_1_through_4(self):
        for io_width in range(1, 4 + 1):
            with self.subTest(io_width=io_width):
                self.tst(io_width)

    def test_5(self):
        self.tst(5)

    def test_6(self):
        self.tst(6)

    def test_sim_5(self):
        self.tst_sim(5)

    def tst_params(self, io_width):
        assert isinstance(io_width, int)
        params = GoldschmidtDivParams.get(io_width)
        print()
        print(params)

    def test_params_1(self):
        self.tst_params(1)

    def test_params_2(self):
        self.tst_params(2)

    def test_params_3(self):
        self.tst_params(3)

    def test_params_4(self):
        self.tst_params(4)

    def test_params_5(self):
        self.tst_params(5)

    def test_params_6(self):
        self.tst_params(6)

    def test_params_7(self):
        self.tst_params(7)

    def test_params_8(self):
        self.tst_params(8)

    def test_params_9(self):
        self.tst_params(9)

    def test_params_10(self):
        self.tst_params(10)

    def test_params_11(self):
        self.tst_params(11)

    def test_params_12(self):
        self.tst_params(12)

    def test_params_13(self):
        self.tst_params(13)

    def test_params_14(self):
        self.tst_params(14)

    def test_params_15(self):
        self.tst_params(15)

    def test_params_16(self):
        self.tst_params(16)

    def test_params_17(self):
        self.tst_params(17)

    def test_params_18(self):
        self.tst_params(18)

    def test_params_19(self):
        self.tst_params(19)

    def test_params_20(self):
        self.tst_params(20)

    def test_params_21(self):
        self.tst_params(21)

    def test_params_22(self):
        self.tst_params(22)

    def test_params_23(self):
        self.tst_params(23)

    def test_params_24(self):
        self.tst_params(24)

    def test_params_25(self):
        self.tst_params(25)

    def test_params_26(self):
        self.tst_params(26)

    def test_params_27(self):
        self.tst_params(27)

    def test_params_28(self):
        self.tst_params(28)

    def test_params_29(self):
        self.tst_params(29)

    def test_params_30(self):
        self.tst_params(30)

    def test_params_31(self):
        self.tst_params(31)

    def test_params_32(self):
        self.tst_params(32)

    def test_params_33(self):
        self.tst_params(33)

    def test_params_34(self):
        self.tst_params(34)

    def test_params_35(self):
        self.tst_params(35)

    def test_params_36(self):
        self.tst_params(36)

    def test_params_37(self):
        self.tst_params(37)

    def test_params_38(self):
        self.tst_params(38)

    def test_params_39(self):
        self.tst_params(39)

    def test_params_40(self):
        self.tst_params(40)

    def test_params_41(self):
        self.tst_params(41)

    def test_params_42(self):
        self.tst_params(42)

    def test_params_43(self):
        self.tst_params(43)

    def test_params_44(self):
        self.tst_params(44)

    def test_params_45(self):
        self.tst_params(45)

    def test_params_46(self):
        self.tst_params(46)

    def test_params_47(self):
        self.tst_params(47)

    def test_params_48(self):
        self.tst_params(48)

    def test_params_49(self):
        self.tst_params(49)

    def test_params_50(self):
        self.tst_params(50)

    def test_params_51(self):
        self.tst_params(51)

    def test_params_52(self):
        self.tst_params(52)

    def test_params_53(self):
        self.tst_params(53)

    def test_params_54(self):
        self.tst_params(54)

    def test_params_55(self):
        self.tst_params(55)

    def test_params_56(self):
        self.tst_params(56)

    def test_params_57(self):
        self.tst_params(57)

    def test_params_58(self):
        self.tst_params(58)

    def test_params_59(self):
        self.tst_params(59)

    def test_params_60(self):
        self.tst_params(60)

    def test_params_61(self):
        self.tst_params(61)

    def test_params_62(self):
        self.tst_params(62)

    def test_params_63(self):
        self.tst_params(63)

    def test_params_64(self):
        self.tst_params(64)


class TestGoldschmidtSqrtRSqrt(FHDLTestCase):
    def tst(self, io_width, frac_wid, extra_precision,
            table_addr_bits, table_data_bits, iter_count):
        assert isinstance(io_width, int)
        assert isinstance(frac_wid, int)
        assert isinstance(extra_precision, int)
        assert isinstance(table_addr_bits, int)
        assert isinstance(table_data_bits, int)
        assert isinstance(iter_count, int)
        with self.subTest(io_width=io_width, frac_wid=frac_wid,
                          extra_precision=extra_precision,
                          table_addr_bits=table_addr_bits,
                          table_data_bits=table_data_bits,
                          iter_count=iter_count):
            for bits in range(1 << io_width):
                radicand = FixedPoint(bits, frac_wid)
                expected_sqrt = radicand.sqrt(RoundDir.DOWN)
                expected_rsqrt = FixedPoint(0, frac_wid)
                if radicand > 0:
                    expected_rsqrt = radicand.rsqrt(RoundDir.DOWN)
                with self.subTest(radicand=repr(radicand),
                                  expected_sqrt=repr(expected_sqrt),
                                  expected_rsqrt=repr(expected_rsqrt)):
                    sqrt, rsqrt = goldschmidt_sqrt_rsqrt(
                        radicand=radicand, io_width=io_width,
                        frac_wid=frac_wid,
                        extra_precision=extra_precision,
                        table_addr_bits=table_addr_bits,
                        table_data_bits=table_data_bits,
                        iter_count=iter_count)
                    with self.subTest(sqrt=repr(sqrt), rsqrt=repr(rsqrt)):
                        self.assertEqual((sqrt, rsqrt),
                                         (expected_sqrt, expected_rsqrt))

    def test1(self):
        self.tst(io_width=16, frac_wid=8, extra_precision=20,
                 table_addr_bits=4, table_data_bits=28, iter_count=4)


if __name__ == "__main__":
    unittest.main()
