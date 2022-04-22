# SPDX-License-Identifier: LGPL-3-or-later
# Copyright 2022 Jacob Lifshay programmerjake@gmail.com

# Funded by NLnet Assure Programme 2021-02-052, https://nlnet.nl/assure part
# of Horizon 2020 EU Programme 957073.

from dataclasses import dataclass
import math
import enum


@enum.unique
class RoundDir(enum.Enum):
    DOWN = enum.auto()
    UP = enum.auto()
    NEAREST_TIES_UP = enum.auto()
    ERROR_IF_INEXACT = enum.auto()


@dataclass(frozen=True)
class FixedPoint:
    bits: int
    frac_wid: int

    def __post_init__(self):
        assert isinstance(self.bits, int)
        assert isinstance(self.frac_wid, int) and self.frac_wid >= 0

    @staticmethod
    def cast(value):
        """convert `value` to a fixed-point number with enough fractional
        bits to preserve its value."""
        if isinstance(value, FixedPoint):
            return value
        if isinstance(value, int):
            return FixedPoint(value, 0)
        if isinstance(value, str):
            value = value.strip()
            neg = value.startswith("-")
            if neg or value.startswith("+"):
                value = value[1:]
            if value.startswith(("0x", "0X")) and "." in value:
                value = value[2:]
                got_dot = False
                bits = 0
                frac_wid = 0
                for digit in value:
                    if digit == "_":
                        continue
                    if got_dot:
                        if digit == ".":
                            raise ValueError("too many `.` in string")
                        frac_wid += 4
                    if digit == ".":
                        got_dot = True
                        continue
                    if not digit.isalnum():
                        raise ValueError("invalid hexadecimal digit")
                    bits <<= 4
                    bits |= int("0x" + digit, base=16)
            else:
                bits = int(value, base=0)
                frac_wid = 0
            if neg:
                bits = -bits
            return FixedPoint(bits, frac_wid)

        if isinstance(value, float):
            n, d = value.as_integer_ratio()
            log2_d = d.bit_length() - 1
            assert d == 1 << log2_d, ("d isn't a power of 2 -- won't ever "
                                      "fail with float being IEEE 754")
            return FixedPoint(n, log2_d)
        raise TypeError("can't convert type to FixedPoint")

    @staticmethod
    def with_frac_wid(value, frac_wid, round_dir=RoundDir.ERROR_IF_INEXACT):
        """convert `value` to the nearest fixed-point number with `frac_wid`
        fractional bits, rounding according to `round_dir`."""
        value = FixedPoint.cast(value)
        assert isinstance(frac_wid, int) and frac_wid >= 0
        assert isinstance(round_dir, RoundDir)
        # compute number of bits that should be removed from value
        del_bits = value.frac_wid - frac_wid
        if del_bits == 0:
            return value
        if del_bits < 0:  # add bits
            return FixedPoint(value.bits << -del_bits,
                              frac_wid)
        if round_dir == RoundDir.DOWN:
            bits = value.bits >> del_bits
        elif round_dir == RoundDir.UP:
            bits = -((-value.bits) >> del_bits)
        elif round_dir == RoundDir.NEAREST_TIES_UP:
            bits = value.bits >> (del_bits - 1)
            bits += 1
            bits >>= 1
        elif round_dir == RoundDir.ERROR_IF_INEXACT:
            bits = value.bits >> del_bits
            if bits << del_bits != value.bits:
                raise ValueError("inexact conversion")
        else:
            assert False, "unimplemented round_dir"
        return FixedPoint(bits, frac_wid)

    def to_frac_wid(self, frac_wid, round_dir=RoundDir.ERROR_IF_INEXACT):
        """convert to the nearest fixed-point number with `frac_wid`
        fractional bits, rounding according to `round_dir`."""
        return FixedPoint.with_frac_wid(self, frac_wid, round_dir)

    def __float__(self):
        return self.bits * 2.0 ** -self.frac_wid

    def cmp(self, rhs):
        """compare self with rhs, returning a positive integer if self is
        greater than rhs, zero if self is equal to rhs, and a negative integer
        if self is less than rhs."""
        rhs = FixedPoint.cast(rhs)
        common_frac_wid = max(self.frac_wid, rhs.frac_wid)
        lhs = self.to_frac_wid(common_frac_wid)
        rhs = rhs.to_frac_wid(common_frac_wid)
        return lhs.bits - rhs.bits

    def __eq__(self, rhs):
        return self.cmp(rhs) == 0

    def __ne__(self, rhs):
        return self.cmp(rhs) != 0

    def __gt__(self, rhs):
        return self.cmp(rhs) > 0

    def __lt__(self, rhs):
        return self.cmp(rhs) < 0

    def __ge__(self, rhs):
        return self.cmp(rhs) >= 0

    def __le__(self, rhs):
        return self.cmp(rhs) <= 0

    def fract(self):
        """return the fractional part of `self`.
        that is `self - math.floor(self)`.
        """
        fract_mask = (1 << self.frac_wid) - 1
        return FixedPoint(self.bits & fract_mask, self.frac_wid)

    def __str__(self):
        if self < 0:
            return "-" + str(-self)
        digit_bits = 4
        frac_digit_count = (self.frac_wid + digit_bits - 1) // digit_bits
        fract = self.fract().to_frac_wid(frac_digit_count * digit_bits)
        frac_str = hex(fract.bits)[2:].zfill(frac_digit_count)
        return hex(math.floor(self)) + "." + frac_str

    def __repr__(self):
        return f"FixedPoint.with_frac_wid({str(self)!r}, {self.frac_wid})"

    def __add__(self, rhs):
        rhs = FixedPoint.cast(rhs)
        common_frac_wid = max(self.frac_wid, rhs.frac_wid)
        lhs = self.to_frac_wid(common_frac_wid)
        rhs = rhs.to_frac_wid(common_frac_wid)
        return FixedPoint(lhs.bits + rhs.bits, common_frac_wid)

    def __neg__(self):
        return FixedPoint(-self.bits, self.frac_wid)

    def __sub__(self, rhs):
        rhs = FixedPoint.cast(rhs)
        common_frac_wid = max(self.frac_wid, rhs.frac_wid)
        lhs = self.to_frac_wid(common_frac_wid)
        rhs = rhs.to_frac_wid(common_frac_wid)
        return FixedPoint(lhs.bits - rhs.bits, common_frac_wid)

    def __mul__(self, rhs):
        rhs = FixedPoint.cast(rhs)
        return FixedPoint(self.bits * rhs.bits, self.frac_wid + rhs.frac_wid)

    def __floor__(self):
        return self.bits >> self.frac_wid


def goldschmidt_div(n, d, width):
    """ Goldschmidt division algorithm.

        based on:
        Even, G., Seidel, P. M., & Ferguson, W. E. (2003).
        A Parametric Error Analysis of Goldschmidt's Division Algorithm.
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.1238&rep=rep1&type=pdf

        arguments:
        n: int
            numerator. a `2*width`-bit unsigned integer.
            must be less than `d << width`, otherwise the quotient wouldn't
            fit in `width` bits.
        d: int
            denominator. a `width`-bit unsigned integer. must not be zero.
        width: int
            the bit-width of the inputs/outputs. must be a positive integer.

        returns: int
            the quotient. a `width`-bit unsigned integer.
    """
    assert isinstance(width, int) and width >= 1
    assert isinstance(d, int) and 0 < d < (1 << width)
    assert isinstance(n, int) and 0 <= n < (d << width)

    # FIXME: calculate best values for extra_precision, table_addr_bits, and
    # table_data_bits -- these are wrong
    extra_precision = width + 3
    table_addr_bits = 4
    table_data_bits = 8

    width += extra_precision

    table = []
    for i in range(1 << table_addr_bits):
        value = 1 / (1 + i * 2 ** -table_addr_bits)
        table.append(FixedPoint.with_frac_wid(value, table_data_bits,
                                              RoundDir.DOWN))

    # this whole algorithm is done with fixed-point arithmetic where values
    # have `width` fractional bits

    n = FixedPoint(n, width)
    d = FixedPoint(d, width)

    # normalize so 1 <= d < 2
    # can easily be done with count-leading-zeros and left shift
    while d < 1:
        n = (n * 2).to_frac_wid(width)
        d = (d * 2).to_frac_wid(width)

    n_shift = 0
    # normalize so 1 <= n < 2
    while n >= 2:
        n = (n * 0.5).to_frac_wid(width)
        n_shift += 1

    # compute initial f by table lookup
    f = table[(d - 1).to_frac_wid(table_addr_bits, RoundDir.DOWN).bits]

    min_bits_of_precision = 1
    while min_bits_of_precision < width * 2:
        # multiply both n and d by f
        n *= f
        d *= f
        n = n.to_frac_wid(width, round_dir=RoundDir.DOWN)
        d = d.to_frac_wid(width, round_dir=RoundDir.UP)

        # slightly less than 2 to make the computation just a bitwise not
        nearly_two = FixedPoint.with_frac_wid(2, width)
        nearly_two = FixedPoint(nearly_two.bits - 1, width)
        f = (nearly_two - d).to_frac_wid(width)

        min_bits_of_precision *= 2

    # scale to correct value
    n *= 1 << n_shift

    # avoid incorrectly rounding down
    n = n.to_frac_wid(width - extra_precision, round_dir=RoundDir.UP)
    return math.floor(n)
