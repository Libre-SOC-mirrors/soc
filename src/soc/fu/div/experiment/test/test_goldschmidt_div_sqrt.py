# SPDX-License-Identifier: LGPL-3-or-later
# Copyright 2022 Jacob Lifshay programmerjake@gmail.com

# Funded by NLnet Assure Programme 2021-02-052, https://nlnet.nl/assure part
# of Horizon 2020 EU Programme 957073.

import unittest
from nmutil.formaltest import FHDLTestCase
from soc.fu.div.experiment.goldschmidt_div_sqrt import (GoldschmidtDivParams, goldschmidt_div,
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
    @unittest.skip("goldschmidt_div isn't finished yet")
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

    def test_1_through_5(self):
        for io_width in range(1, 5 + 1):
            with self.subTest(io_width=io_width):
                self.tst(io_width)

    def test_6(self):
        self.tst(6)


if __name__ == "__main__":
    unittest.main()
