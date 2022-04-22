# SPDX-License-Identifier: LGPL-3-or-later
# Copyright 2022 Jacob Lifshay programmerjake@gmail.com

# Funded by NLnet Assure Programme 2021-02-052, https://nlnet.nl/assure part
# of Horizon 2020 EU Programme 957073.

import unittest
from nmutil.formaltest import FHDLTestCase
from soc.fu.div.experiment.goldschmidt_div_sqrt import (goldschmidt_div,
                                                        FixedPoint)


class TestFixedPoint(FHDLTestCase):
    def test_str_roundtrip(self):
        for frac_wid in range(8):
            for bits in range(-1 << 9, 1 << 9):
                with self.subTest(bits=hex(bits), frac_wid=frac_wid):
                    value = FixedPoint(bits, frac_wid)
                    round_trip_value = FixedPoint.cast(str(value))
                    self.assertEqual(value, round_trip_value)


class TestGoldschmidtDiv(FHDLTestCase):
    def tst(self, width):
        assert isinstance(width, int)
        for d in range(1, 1 << width):
            for n in range(d << width):
                expected = n // d
                with self.subTest(width=width, n=hex(n), d=hex(d),
                                  expected=hex(expected)):
                    result = goldschmidt_div(n, d, width)
                    self.assertEqual(result, expected, f"result={hex(result)}")

    def test_1_through_5(self):
        for width in range(1, 5 + 1):
            self.tst(width)

    def test_6(self):
        self.tst(6)


if __name__ == "__main__":
    unittest.main()
