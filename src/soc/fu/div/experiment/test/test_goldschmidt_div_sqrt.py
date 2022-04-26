# SPDX-License-Identifier: LGPL-3-or-later
# Copyright 2022 Jacob Lifshay programmerjake@gmail.com

# Funded by NLnet Assure Programme 2021-02-052, https://nlnet.nl/assure part
# of Horizon 2020 EU Programme 957073.

import unittest
from nmutil.formaltest import FHDLTestCase
from soc.fu.div.experiment.goldschmidt_div_sqrt import (
    GoldschmidtDivParams, ParamsNotAccurateEnough, goldschmidt_div, FixedPoint)


class TestFixedPoint(FHDLTestCase):
    def test_str_roundtrip(self):
        for frac_wid in range(8):
            for bits in range(-1 << 9, 1 << 9):
                with self.subTest(bits=hex(bits), frac_wid=frac_wid):
                    value = FixedPoint(bits, frac_wid)
                    round_trip_value = FixedPoint.cast(str(value))
                    self.assertEqual(value, round_trip_value)


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

    def test_1_through_4(self):
        for io_width in range(1, 4 + 1):
            with self.subTest(io_width=io_width):
                self.tst(io_width)

    def test_5(self):
        self.tst(5)

    def test_6(self):
        self.tst(6)

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


if __name__ == "__main__":
    unittest.main()
