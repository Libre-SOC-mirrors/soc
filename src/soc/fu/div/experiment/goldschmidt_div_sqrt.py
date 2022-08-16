# SPDX-License-Identifier: LGPL-3-or-later
# Copyright 2022 Jacob Lifshay programmerjake@gmail.com

# Funded by NLnet Assure Programme 2021-02-052, https://nlnet.nl/assure part
# of Horizon 2020 EU Programme 957073.

from collections import defaultdict
import logging
import math
import enum
from fractions import Fraction
from types import FunctionType
from functools import lru_cache
from nmigen.hdl.ast import Signal, unsigned, signed, Const
from nmigen.hdl.dsl import Module, Elaboratable
from nmigen.hdl.mem import Memory
from nmutil.clz import CLZ
from nmutil.plain_data import plain_data, fields, replace

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

# fix broken IDE type detection for cached_property
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from functools import cached_property


_NOT_FOUND = object()


def cache_on_self(func):
    """like `functools.cached_property`, except for methods. unlike
    `lru_cache` the cache is per-class instance rather than a global cache
    per-method."""

    assert isinstance(func, FunctionType), \
        "non-plain methods are not supported"

    cache_name = func.__name__ + "__cache"

    def wrapper(self, *args, **kwargs):
        # specifically access through `__dict__` to bypass frozen=True
        cache = self.__dict__.get(cache_name, _NOT_FOUND)
        if cache is _NOT_FOUND:
            self.__dict__[cache_name] = cache = {}
        key = (args, *kwargs.items())
        retval = cache.get(key, _NOT_FOUND)
        if retval is _NOT_FOUND:
            retval = func(self, *args, **kwargs)
            cache[key] = retval
        return retval

    wrapper.__doc__ = func.__doc__
    return wrapper


@enum.unique
class RoundDir(enum.Enum):
    DOWN = enum.auto()
    UP = enum.auto()
    NEAREST_TIES_UP = enum.auto()
    ERROR_IF_INEXACT = enum.auto()


@plain_data(frozen=True, eq=False, repr=False)
class FixedPoint:
    __slots__ = "bits", "frac_wid"

    def __init__(self, bits, frac_wid):
        self.bits = bits
        self.frac_wid = frac_wid
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
        assert isinstance(frac_wid, int) and frac_wid >= 0
        assert isinstance(round_dir, RoundDir)
        if isinstance(value, Fraction):
            numerator = value.numerator
            denominator = value.denominator
        else:
            value = FixedPoint.cast(value)
            numerator = value.bits
            denominator = 1 << value.frac_wid
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        bits, remainder = divmod(numerator << frac_wid, denominator)
        if round_dir == RoundDir.DOWN:
            pass
        elif round_dir == RoundDir.UP:
            if remainder != 0:
                bits += 1
        elif round_dir == RoundDir.NEAREST_TIES_UP:
            if remainder * 2 >= denominator:
                bits += 1
        elif round_dir == RoundDir.ERROR_IF_INEXACT:
            if remainder != 0:
                raise ValueError("inexact conversion")
        else:
            assert False, "unimplemented round_dir"
        return FixedPoint(bits, frac_wid)

    def to_frac_wid(self, frac_wid, round_dir=RoundDir.ERROR_IF_INEXACT):
        """convert to the nearest fixed-point number with `frac_wid`
        fractional bits, rounding according to `round_dir`."""
        return FixedPoint.with_frac_wid(self, frac_wid, round_dir)

    def __float__(self):
        # use truediv to get correct result even when bits
        # and frac_wid are huge
        return float(self.bits / (1 << self.frac_wid))

    def as_fraction(self):
        return Fraction(self.bits, 1 << self.frac_wid)

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

    def __radd__(self, lhs):
        # symmetric
        return self.__add__(lhs)

    def __neg__(self):
        return FixedPoint(-self.bits, self.frac_wid)

    def __sub__(self, rhs):
        rhs = FixedPoint.cast(rhs)
        common_frac_wid = max(self.frac_wid, rhs.frac_wid)
        lhs = self.to_frac_wid(common_frac_wid)
        rhs = rhs.to_frac_wid(common_frac_wid)
        return FixedPoint(lhs.bits - rhs.bits, common_frac_wid)

    def __rsub__(self, lhs):
        # a - b == -(b - a)
        return -self.__sub__(lhs)

    def __mul__(self, rhs):
        rhs = FixedPoint.cast(rhs)
        return FixedPoint(self.bits * rhs.bits, self.frac_wid + rhs.frac_wid)

    def __rmul__(self, lhs):
        # symmetric
        return self.__mul__(lhs)

    def __floor__(self):
        return self.bits >> self.frac_wid

    def div(self, rhs, frac_wid, round_dir=RoundDir.ERROR_IF_INEXACT):
        assert isinstance(frac_wid, int) and frac_wid >= 0
        assert isinstance(round_dir, RoundDir)
        rhs = FixedPoint.cast(rhs)
        return FixedPoint.with_frac_wid(self.as_fraction()
                                        / rhs.as_fraction(),
                                        frac_wid, round_dir)

    def sqrt(self, round_dir=RoundDir.ERROR_IF_INEXACT):
        assert isinstance(round_dir, RoundDir)
        if self < 0:
            raise ValueError("can't compute sqrt of negative number")
        if self == 0:
            return self
        retval = FixedPoint(0, self.frac_wid)
        int_part_wid = self.bits.bit_length() - self.frac_wid
        first_bit_index = -(-int_part_wid // 2)  # division rounds up
        last_bit_index = -self.frac_wid
        for bit_index in range(first_bit_index, last_bit_index - 1, -1):
            trial = retval + FixedPoint(1 << (bit_index + self.frac_wid),
                                        self.frac_wid)
            if trial * trial <= self:
                retval = trial
        if round_dir == RoundDir.DOWN:
            pass
        elif round_dir == RoundDir.UP:
            if retval * retval < self:
                retval += FixedPoint(1, self.frac_wid)
        elif round_dir == RoundDir.NEAREST_TIES_UP:
            half_way = retval + FixedPoint(1, self.frac_wid + 1)
            if half_way * half_way <= self:
                retval += FixedPoint(1, self.frac_wid)
        elif round_dir == RoundDir.ERROR_IF_INEXACT:
            if retval * retval != self:
                raise ValueError("inexact sqrt")
        else:
            assert False, "unimplemented round_dir"
        return retval

    def rsqrt(self, round_dir=RoundDir.ERROR_IF_INEXACT):
        """compute the reciprocal-sqrt of `self`"""
        assert isinstance(round_dir, RoundDir)
        if self < 0:
            raise ValueError("can't compute rsqrt of negative number")
        if self == 0:
            raise ZeroDivisionError("can't compute rsqrt of zero")
        retval = FixedPoint(0, self.frac_wid)
        first_bit_index = -(-self.frac_wid // 2)  # division rounds up
        last_bit_index = -self.frac_wid
        for bit_index in range(first_bit_index, last_bit_index - 1, -1):
            trial = retval + FixedPoint(1 << (bit_index + self.frac_wid),
                                        self.frac_wid)
            if trial * trial * self <= 1:
                retval = trial
        if round_dir == RoundDir.DOWN:
            pass
        elif round_dir == RoundDir.UP:
            if retval * retval * self < 1:
                retval += FixedPoint(1, self.frac_wid)
        elif round_dir == RoundDir.NEAREST_TIES_UP:
            half_way = retval + FixedPoint(1, self.frac_wid + 1)
            if half_way * half_way * self <= 1:
                retval += FixedPoint(1, self.frac_wid)
        elif round_dir == RoundDir.ERROR_IF_INEXACT:
            if retval * retval * self != 1:
                raise ValueError("inexact rsqrt")
        else:
            assert False, "unimplemented round_dir"
        return retval


class ParamsNotAccurateEnough(Exception):
    """raised when the parameters aren't accurate enough to have goldschmidt
    division work."""


def _assert_accuracy(condition, msg="not accurate enough"):
    if condition:
        return
    raise ParamsNotAccurateEnough(msg)


@plain_data(frozen=True, unsafe_hash=True)
class GoldschmidtDivParamsBase:
    """parameters for a Goldschmidt division algorithm, excluding derived
    parameters.
    """

    __slots__ = ("io_width", "extra_precision", "table_addr_bits",
                 "table_data_bits", "iter_count")

    def __init__(self, io_width, extra_precision, table_addr_bits,
                 table_data_bits, iter_count):
        assert isinstance(io_width, int)
        assert isinstance(extra_precision, int)
        assert isinstance(table_addr_bits, int)
        assert isinstance(table_data_bits, int)
        assert isinstance(iter_count, int)
        self.io_width = io_width
        """bit-width of the input divisor and the result.
        the input numerator is `2 * io_width`-bits wide.
        """

        self.extra_precision = extra_precision
        """number of bits of additional precision used inside the algorithm."""

        self.table_addr_bits = table_addr_bits
        """the number of address bits used in the lookup-table."""

        self.table_data_bits = table_data_bits
        """the number of data bits used in the lookup-table."""

        self.iter_count = iter_count
        """the total number of iterations of the division algorithm's loop"""


@plain_data(frozen=True, unsafe_hash=True)
class GoldschmidtDivParams(GoldschmidtDivParamsBase):
    """parameters for a Goldschmidt division algorithm.
    Use `GoldschmidtDivParams.get` to find a efficient set of parameters.
    """

    __slots__ = "table", "ops"

    def _shrink_bound(self, bound, round_dir):
        """prevent fractions from having huge numerators/denominators by
        rounding to a `FixedPoint` and converting back to a `Fraction`.

        This is intended only for values used to compute bounds, and not for
        values that end up in the hardware.
        """
        assert isinstance(bound, (Fraction, int))
        assert round_dir is RoundDir.DOWN or round_dir is RoundDir.UP, \
            "you shouldn't use that round_dir on bounds"
        frac_wid = self.io_width * 4 + 100  # should be enough precision
        fixed = FixedPoint.with_frac_wid(bound, frac_wid, round_dir)
        return fixed.as_fraction()

    def _shrink_min(self, min_bound):
        """prevent fractions used as minimum bounds from having huge
        numerators/denominators by rounding down to a `FixedPoint` and
        converting back to a `Fraction`.

        This is intended only for values used to compute bounds, and not for
        values that end up in the hardware.
        """
        return self._shrink_bound(min_bound, RoundDir.DOWN)

    def _shrink_max(self, max_bound):
        """prevent fractions used as maximum bounds from having huge
        numerators/denominators by rounding up to a `FixedPoint` and
        converting back to a `Fraction`.

        This is intended only for values used to compute bounds, and not for
        values that end up in the hardware.
        """
        return self._shrink_bound(max_bound, RoundDir.UP)

    @property
    def table_addr_count(self):
        """number of distinct addresses in the lookup-table."""
        # used while computing self.table, so can't just do len(self.table)
        return 1 << self.table_addr_bits

    def table_input_exact_range(self, addr):
        """return the range of inputs as `Fraction`s used for the table entry
        with address `addr`."""
        assert isinstance(addr, int)
        assert 0 <= addr < self.table_addr_count
        _assert_accuracy(self.io_width >= self.table_addr_bits)
        addr_shift = self.io_width - self.table_addr_bits
        min_numerator = (1 << self.io_width) + (addr << addr_shift)
        denominator = 1 << self.io_width
        values_per_table_entry = 1 << addr_shift
        max_numerator = min_numerator + values_per_table_entry - 1
        min_input = Fraction(min_numerator, denominator)
        max_input = Fraction(max_numerator, denominator)
        min_input = self._shrink_min(min_input)
        max_input = self._shrink_max(max_input)
        assert 1 <= min_input <= max_input < 2
        return min_input, max_input

    def table_value_exact_range(self, addr):
        """return the range of values as `Fraction`s used for the table entry
        with address `addr`."""
        min_input, max_input = self.table_input_exact_range(addr)
        # division swaps min/max
        min_value = 1 / max_input
        max_value = 1 / min_input
        min_value = self._shrink_min(min_value)
        max_value = self._shrink_max(max_value)
        assert 0.5 < min_value <= max_value <= 1
        return min_value, max_value

    def table_exact_value(self, index):
        min_value, max_value = self.table_value_exact_range(index)
        # we round down
        return min_value

    def __init__(self, io_width, extra_precision, table_addr_bits,
                 table_data_bits, iter_count):
        super().__init__(io_width=io_width,
                         extra_precision=extra_precision,
                         table_addr_bits=table_addr_bits,
                         table_data_bits=table_data_bits,
                         iter_count=iter_count)
        _assert_accuracy(self.io_width >= 1, "io_width out of range")
        _assert_accuracy(self.extra_precision >= 0,
                         "extra_precision out of range")
        _assert_accuracy(self.table_addr_bits >= 1,
                         "table_addr_bits out of range")
        _assert_accuracy(self.table_data_bits >= 1,
                         "table_data_bits out of range")
        _assert_accuracy(self.iter_count >= 1, "iter_count out of range")
        table = []
        for addr in range(1 << self.table_addr_bits):
            table.append(FixedPoint.with_frac_wid(self.table_exact_value(addr),
                                                  self.table_data_bits,
                                                  RoundDir.DOWN))

        self.table = tuple(table)
        """ the lookup-table.
        type: tuple[FixedPoint, ...]
        """

        self.ops = tuple(self.__make_ops())
        "the operations needed to perform the goldschmidt division algorithm."

    @property
    def expanded_width(self):
        """the total number of bits of precision used inside the algorithm."""
        return self.io_width + self.extra_precision

    @property
    def n_d_f_int_wid(self):
        """the number of bits in the integer part of `state.n`, `state.d`, and
        `state.f` during the main iteration loop.
        """
        return 2

    @property
    def n_d_f_total_wid(self):
        """the total number of bits (both integer and fraction bits) in
        `state.n`, `state.d`, and `state.f` during the main iteration loop.
        """
        return self.n_d_f_int_wid + self.expanded_width

    @cache_on_self
    def max_neps(self, i):
        """maximum value of `neps[i]`.
        `neps[i]` is defined to be `n[i] * N_prime[i - 1] * F_prime[i - 1]`.
        """
        assert isinstance(i, int) and 0 <= i < self.iter_count
        return Fraction(1, 1 << self.expanded_width)

    @cache_on_self
    def max_deps(self, i):
        """maximum value of `deps[i]`.
        `deps[i]` is defined to be `d[i] * D_prime[i - 1] * F_prime[i - 1]`.
        """
        assert isinstance(i, int) and 0 <= i < self.iter_count
        return Fraction(1, 1 << self.expanded_width)

    @cache_on_self
    def max_feps(self, i):
        """maximum value of `feps[i]`.
        `feps[i]` is defined to be `f[i] * (2 - D_prime[i - 1])`.
        """
        assert isinstance(i, int) and 0 <= i < self.iter_count
        # zero, because the computation of `F_prime[i]` in
        # `GoldschmidtDivOp.MulDByF.run(...)` is exact.
        return Fraction(0)

    @cached_property
    def e0_range(self):
        """minimum and maximum values of `e[0]`
        (the relative error in `F_prime[-1]`)
        """
        min_e0 = Fraction(0)
        max_e0 = Fraction(0)
        for addr in range(self.table_addr_count):
            # `F_prime[-1] = (1 - e[0]) / B`
            # => `e[0] = 1 - B * F_prime[-1]`
            min_b, max_b = self.table_input_exact_range(addr)
            f_prime_m1 = self.table[addr].as_fraction()
            assert min_b >= 0 and f_prime_m1 >= 0, \
                "only positive quadrant of interval multiplication implemented"
            min_product = min_b * f_prime_m1
            max_product = max_b * f_prime_m1
            # negation swaps min/max
            cur_min_e0 = 1 - max_product
            cur_max_e0 = 1 - min_product
            min_e0 = min(min_e0, cur_min_e0)
            max_e0 = max(max_e0, cur_max_e0)
        min_e0 = self._shrink_min(min_e0)
        max_e0 = self._shrink_max(max_e0)
        return min_e0, max_e0

    @cached_property
    def min_e0(self):
        """minimum value of `e[0]` (the relative error in `F_prime[-1]`)
        """
        min_e0, max_e0 = self.e0_range
        return min_e0

    @cached_property
    def max_e0(self):
        """maximum value of `e[0]` (the relative error in `F_prime[-1]`)
        """
        min_e0, max_e0 = self.e0_range
        return max_e0

    @cached_property
    def max_abs_e0(self):
        """maximum value of `abs(e[0])`."""
        return max(abs(self.min_e0), abs(self.max_e0))

    @cached_property
    def min_abs_e0(self):
        """minimum value of `abs(e[0])`."""
        return Fraction(0)

    @cache_on_self
    def max_n(self, i):
        """maximum value of `n[i]` (the relative error in `N_prime[i]`
        relative to the previous iteration)
        """
        assert isinstance(i, int) and 0 <= i < self.iter_count
        if i == 0:
            # from Claim 10
            # `n[0] = neps[0] / ((1 - e[0]) * (A / B))`
            # `n[0] <= 2 * neps[0] / (1 - e[0])`

            assert self.max_e0 < 1 and self.max_neps(0) >= 0, \
                "only one quadrant of interval division implemented"
            retval = 2 * self.max_neps(0) / (1 - self.max_e0)
        elif i == 1:
            # from Claim 10
            # `n[1] <= neps[1] / ((1 - f[0]) * (1 - pi[0] - delta[0]))`
            min_mpd = 1 - self.max_pi(0) - self.max_delta(0)
            assert self.max_f(0) <= 1 and min_mpd >= 0, \
                "only one quadrant of interval multiplication implemented"
            prod = (1 - self.max_f(0)) * min_mpd
            assert self.max_neps(1) >= 0 and prod > 0, \
                "only one quadrant of interval division implemented"
            retval = self.max_neps(1) / prod
        else:
            # from Claim 6
            # `0 <= n[i] <= 2 * max_neps[i] / (1 - pi[i - 1] - delta[i - 1])`
            min_mpd = 1 - self.max_pi(i - 1) - self.max_delta(i - 1)
            assert self.max_neps(i) >= 0 and min_mpd > 0, \
                "only one quadrant of interval division implemented"
            retval = self.max_neps(i) / min_mpd

        return self._shrink_max(retval)

    @cache_on_self
    def max_d(self, i):
        """maximum value of `d[i]` (the relative error in `D_prime[i]`
        relative to the previous iteration)
        """
        assert isinstance(i, int) and 0 <= i < self.iter_count
        if i == 0:
            # from Claim 10
            # `d[0] = deps[0] / (1 - e[0])`

            assert self.max_e0 < 1 and self.max_deps(0) >= 0, \
                "only one quadrant of interval division implemented"
            retval = self.max_deps(0) / (1 - self.max_e0)
        elif i == 1:
            # from Claim 10
            # `d[1] <= deps[1] / ((1 - f[0]) * (1 - delta[0] ** 2))`
            assert self.max_f(0) <= 1 and self.max_delta(0) <= 1, \
                "only one quadrant of interval multiplication implemented"
            divisor = (1 - self.max_f(0)) * (1 - self.max_delta(0) ** 2)
            assert self.max_deps(1) >= 0 and divisor > 0, \
                "only one quadrant of interval division implemented"
            retval = self.max_deps(1) / divisor
        else:
            # from Claim 6
            # `0 <= d[i] <= max_deps[i] / (1 - delta[i - 1])`
            assert self.max_deps(i) >= 0 and self.max_delta(i - 1) < 1, \
                "only one quadrant of interval division implemented"
            retval = self.max_deps(i) / (1 - self.max_delta(i - 1))

        return self._shrink_max(retval)

    @cache_on_self
    def max_f(self, i):
        """maximum value of `f[i]` (the relative error in `F_prime[i]`
        relative to the previous iteration)
        """
        assert isinstance(i, int) and 0 <= i < self.iter_count
        if i == 0:
            # from Claim 10
            # `f[0] = feps[0] / (1 - delta[0])`

            assert self.max_delta(0) < 1 and self.max_feps(0) >= 0, \
                "only one quadrant of interval division implemented"
            retval = self.max_feps(0) / (1 - self.max_delta(0))
        elif i == 1:
            # from Claim 10
            # `f[1] = feps[1]`
            retval = self.max_feps(1)
        else:
            # from Claim 6
            # `f[i] <= max_feps[i]`
            retval = self.max_feps(i)

        return self._shrink_max(retval)

    @cache_on_self
    def max_delta(self, i):
        """ maximum value of `delta[i]`.
        `delta[i]` is defined in Definition 4 of paper.
        """
        assert isinstance(i, int) and 0 <= i < self.iter_count
        if i == 0:
            # `delta[0] = abs(e[0]) + 3 * d[0] / 2`
            retval = self.max_abs_e0 + Fraction(3, 2) * self.max_d(0)
        else:
            # `delta[i] = delta[i - 1] ** 2 + f[i - 1]`
            prev_max_delta = self.max_delta(i - 1)
            assert prev_max_delta >= 0
            retval = prev_max_delta ** 2 + self.max_f(i - 1)

        # `delta[i]` has to be smaller than one otherwise errors would go off
        # to infinity
        _assert_accuracy(retval < 1)

        return self._shrink_max(retval)

    @cache_on_self
    def max_pi(self, i):
        """ maximum value of `pi[i]`.
        `pi[i]` is defined right below Theorem 5 of paper.
        """
        assert isinstance(i, int) and 0 <= i < self.iter_count
        # `pi[i] = 1 - (1 - n[i]) * prod`
        # where `prod` is the product of,
        # for `j` in `0 <= j < i`, `(1 - n[j]) / (1 + d[j])`
        min_prod = Fraction(1)
        for j in range(i):
            max_n_j = self.max_n(j)
            max_d_j = self.max_d(j)
            assert max_n_j <= 1 and max_d_j > -1, \
                "only one quadrant of interval division implemented"
            min_prod *= (1 - max_n_j) / (1 + max_d_j)
        max_n_i = self.max_n(i)
        assert max_n_i <= 1 and min_prod >= 0, \
            "only one quadrant of interval multiplication implemented"
        retval = 1 - (1 - max_n_i) * min_prod
        return self._shrink_max(retval)

    @cached_property
    def max_n_shift(self):
        """ maximum value of `state.n_shift`.
        """
        # numerator must be less than `denominator << self.io_width`, so
        # `n_shift` is at most `self.io_width`
        return self.io_width

    @cached_property
    def n_hat(self):
        """ maximum value of, for all `i`, `max_n(i)` and `max_d(i)`
        """
        n_hat = Fraction(0)
        for i in range(self.iter_count):
            n_hat = max(n_hat, self.max_n(i), self.max_d(i))
        return self._shrink_max(n_hat)

    def __make_ops(self):
        """ Goldschmidt division algorithm.

            based on:
            Even, G., Seidel, P. M., & Ferguson, W. E. (2003).
            A Parametric Error Analysis of Goldschmidt's Division Algorithm.
            https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.1238&rep=rep1&type=pdf

            yields: GoldschmidtDivOp
                the operations needed to perform the division.
        """
        # establish assumptions of the paper's error analysis (section 3.1):

        # 1. normalize so A (numerator) and B (denominator) are in [1, 2)
        yield GoldschmidtDivOp.Normalize

        # 2. ensure all relative errors from directed rounding are <= 1 / 4.
        # the assumption is met by multipliers with > 4-bits precision
        _assert_accuracy(self.expanded_width > 4)

        # 3. require `abs(e[0]) + 3 * d[0] / 2 + f[0] < 1 / 2`.
        _assert_accuracy(self.max_abs_e0 + 3 * self.max_d(0) / 2
                         + self.max_f(0) < Fraction(1, 2))

        # 4. the initial approximation F'[-1] of 1/B is in [1/2, 1].
        # (B is the denominator)

        for addr in range(self.table_addr_count):
            f_prime_m1 = self.table[addr]
            _assert_accuracy(0.5 <= f_prime_m1 <= 1)

        yield GoldschmidtDivOp.FEqTableLookup

        # we use Setting I (section 4.1 of the paper):
        # Require `n[i] <= n_hat` and `d[i] <= n_hat` and `f[i] = 0`:
        # the conditions on n_hat are satisfied by construction.
        for i in range(self.iter_count):
            _assert_accuracy(self.max_f(i) == 0)
            yield GoldschmidtDivOp.MulNByF
            if i != self.iter_count - 1:
                yield GoldschmidtDivOp.MulDByF
                yield GoldschmidtDivOp.FEq2MinusD

        # relative approximation error `p(N_prime[i])`:
        # `p(N_prime[i]) = (A / B - N_prime[i]) / (A / B)`
        # `0 <= p(N_prime[i])`
        # `p(N_prime[i]) <= (2 * i) * n_hat \`
        # ` + (abs(e[0]) + 3 * n_hat / 2) ** (2 ** i)`
        i = self.iter_count - 1  # last used `i`
        # compute power manually to prevent huge intermediate values
        power = self._shrink_max(self.max_abs_e0 + 3 * self.n_hat / 2)
        for _ in range(i):
            power = self._shrink_max(power * power)

        max_rel_error = (2 * i) * self.n_hat + power

        min_a_over_b = Fraction(1, 2)
        min_abs_error_for_correctness = min_a_over_b / (1 << self.max_n_shift)
        min_rel_error_for_correctness = (min_abs_error_for_correctness
                                         / min_a_over_b)

        _assert_accuracy(
            max_rel_error < min_rel_error_for_correctness,
            f"not accurate enough: max_rel_error={max_rel_error}"
            f" min_rel_error_for_correctness={min_rel_error_for_correctness}")

        yield GoldschmidtDivOp.CalcResult

    @cache_on_self
    def default_cost_fn(self):
        """ calculate the estimated cost on an arbitrary scale of implementing
        goldschmidt division with the specified parameters. larger cost
        values mean worse parameters.

        This is the default cost function for `GoldschmidtDivParams.get`.

        returns: float
        """
        rom_cells = self.table_data_bits << self.table_addr_bits
        cost = float(rom_cells)
        for op in self.ops:
            if op == GoldschmidtDivOp.MulNByF \
                    or op == GoldschmidtDivOp.MulDByF:
                mul_cost = self.expanded_width ** 2
                mul_cost *= self.expanded_width.bit_length()
                cost += mul_cost
        cost += 5e7 * self.iter_count
        return cost

    @staticmethod
    @lru_cache(maxsize=1 << 16)
    def __cached_new(base_params):
        assert isinstance(base_params, GoldschmidtDivParamsBase)
        kwargs = {}
        for field in fields(GoldschmidtDivParamsBase):
            kwargs[field] = getattr(base_params, field)
        try:
            return GoldschmidtDivParams(**kwargs), None
        except ParamsNotAccurateEnough as e:
            return None, e

    @staticmethod
    def __raise(e):  # type: (ParamsNotAccurateEnough) -> Any
        raise e

    @staticmethod
    def cached_new(base_params, handle_error=__raise):
        assert isinstance(base_params, GoldschmidtDivParamsBase)
        params, error = GoldschmidtDivParams.__cached_new(base_params)
        if error is None:
            return params
        else:
            return handle_error(error)

    @staticmethod
    def get(io_width, cost_fn=default_cost_fn, max_table_addr_bits=12):
        """ find efficient parameters for a goldschmidt division algorithm
        with `params.io_width == io_width`.

        arguments:
        io_width: int
            bit-width of the input divisor and the result.
            the input numerator is `2 * io_width`-bits wide.
        cost_fn: Callable[[GoldschmidtDivParams], float]
            return the estimated cost on an arbitrary scale of implementing
            goldschmidt division with the specified parameters. larger cost
            values mean worse parameters.
        max_table_addr_bits: int
            maximum allowable value of `table_addr_bits`
        """
        assert isinstance(io_width, int) and io_width >= 1
        assert callable(cost_fn)

        last_error = None
        last_error_params = None

        def cached_new(base_params):
            def handle_error(e):
                nonlocal last_error, last_error_params
                last_error = e
                last_error_params = base_params
                return None

            retval = GoldschmidtDivParams.cached_new(base_params, handle_error)
            if retval is None:
                logging.debug(f"GoldschmidtDivParams.get: err: {base_params}")
            else:
                logging.debug(f"GoldschmidtDivParams.get: ok: {base_params}")
            return retval

        @lru_cache(maxsize=None)
        def get_cost(base_params):
            params = cached_new(base_params)
            if params is None:
                return math.inf
            retval = cost_fn(params)
            logging.debug(f"GoldschmidtDivParams.get: cost={retval}: {params}")
            return retval

        # start with parameters big enough to always work.
        initial_extra_precision = io_width * 2 + 4
        initial_params = GoldschmidtDivParamsBase(
            io_width=io_width,
            extra_precision=initial_extra_precision,
            table_addr_bits=min(max_table_addr_bits, io_width),
            table_data_bits=io_width + initial_extra_precision,
            iter_count=1 + io_width.bit_length())

        if cached_new(initial_params) is None:
            raise ValueError(f"initial goldschmidt division algorithm "
                             f"parameters are invalid: {initial_params}"
                             ) from last_error

        # find good initial `iter_count`
        params = initial_params
        for iter_count in range(1, initial_params.iter_count):
            trial_params = replace(params, iter_count=iter_count)
            if cached_new(trial_params) is not None:
                params = trial_params
                break

        # now find `table_addr_bits`
        cost = get_cost(params)
        for table_addr_bits in range(1, max_table_addr_bits):
            trial_params = replace(params, table_addr_bits=table_addr_bits)
            trial_cost = get_cost(trial_params)
            if trial_cost < cost:
                params = trial_params
                cost = trial_cost
                break

        # check one higher `iter_count` to see if it has lower cost
        for table_addr_bits in range(1, max_table_addr_bits + 1):
            trial_params = replace(params,
                                   table_addr_bits=table_addr_bits,
                                   iter_count=params.iter_count + 1)
            trial_cost = get_cost(trial_params)
            if trial_cost < cost:
                params = trial_params
                cost = trial_cost
                break

        # now shrink `table_data_bits`
        while True:
            trial_params = replace(params,
                                   table_data_bits=params.table_data_bits - 1)
            trial_cost = get_cost(trial_params)
            if trial_cost < cost:
                params = trial_params
                cost = trial_cost
            else:
                break

        # and shrink `extra_precision`
        while True:
            trial_params = replace(params,
                                   extra_precision=params.extra_precision - 1)
            trial_cost = get_cost(trial_params)
            if trial_cost < cost:
                params = trial_params
                cost = trial_cost
            else:
                break

        retval = cached_new(params)
        assert isinstance(retval, GoldschmidtDivParams)
        return retval


def clz(v, wid):
    """count leading zeros -- handy for debugging."""
    assert isinstance(wid, int)
    assert isinstance(v, int) and 0 <= v < (1 << wid)
    return (1 << wid).bit_length() - v.bit_length()


@enum.unique
class GoldschmidtDivOp(enum.Enum):
    Normalize = "n, d, n_shift = normalize(n, d)"
    FEqTableLookup = "f = table_lookup(d)"
    MulNByF = "n *= f"
    MulDByF = "d *= f"
    FEq2MinusD = "f = 2 - d"
    CalcResult = "result = unnormalize_and_round(n)"

    def run(self, params, state):
        assert isinstance(params, GoldschmidtDivParams)
        assert isinstance(state, GoldschmidtDivState)
        expanded_width = params.expanded_width
        table_addr_bits = params.table_addr_bits
        if self == GoldschmidtDivOp.Normalize:
            # normalize so 1 <= d < 2
            # can easily be done with count-leading-zeros and left shift
            while state.d < 1:
                state.n = (state.n * 2).to_frac_wid(expanded_width)
                state.d = (state.d * 2).to_frac_wid(expanded_width)

            state.n_shift = 0
            # normalize so 1 <= n < 2
            while state.n >= 2:
                state.n = (state.n * 0.5).to_frac_wid(expanded_width,
                                                      round_dir=RoundDir.DOWN)
                state.n_shift += 1
        elif self == GoldschmidtDivOp.FEqTableLookup:
            # compute initial f by table lookup
            d_m_1 = state.d - 1
            d_m_1 = d_m_1.to_frac_wid(table_addr_bits, RoundDir.DOWN)
            assert 0 <= d_m_1.bits < (1 << params.table_addr_bits)
            state.f = params.table[d_m_1.bits]
            state.f = state.f.to_frac_wid(expanded_width,
                                          round_dir=RoundDir.DOWN)
        elif self == GoldschmidtDivOp.MulNByF:
            assert state.f is not None
            n = state.n * state.f
            state.n = n.to_frac_wid(expanded_width, round_dir=RoundDir.DOWN)
        elif self == GoldschmidtDivOp.MulDByF:
            assert state.f is not None
            d = state.d * state.f
            state.d = d.to_frac_wid(expanded_width, round_dir=RoundDir.UP)
        elif self == GoldschmidtDivOp.FEq2MinusD:
            state.f = (2 - state.d).to_frac_wid(expanded_width)
        elif self == GoldschmidtDivOp.CalcResult:
            assert state.n_shift is not None
            # scale to correct value
            n = state.n * (1 << state.n_shift)

            state.quotient = math.floor(n)
            state.remainder = state.orig_n - state.quotient * state.orig_d
            if state.remainder >= state.orig_d:
                state.quotient += 1
                state.remainder -= state.orig_d
        else:
            assert False, f"unimplemented GoldschmidtDivOp: {self}"

    def gen_hdl(self, params, state, sync_rom):
        """generate the hdl for this operation.

        arguments:
        params: GoldschmidtDivParams
            the goldschmidt division parameters.
        state: GoldschmidtDivHDLState
            the input/output state
        sync_rom: bool
            true if the rom should be read synchronously rather than
            combinatorially, incurring an extra clock cycle of latency.
        """
        assert isinstance(params, GoldschmidtDivParams)
        assert isinstance(state, GoldschmidtDivHDLState)
        m = state.m
        if self == GoldschmidtDivOp.Normalize:
            # normalize so 1 <= d < 2
            assert state.d.width == params.io_width
            assert state.n.width == 2 * params.io_width
            d_leading_zeros = CLZ(params.io_width)
            m.submodules.d_leading_zeros = d_leading_zeros
            m.d.comb += d_leading_zeros.sig_in.eq(state.d)
            d_shift_out = Signal.like(state.d)
            m.d.comb += d_shift_out.eq(state.d << d_leading_zeros.lz)
            d = Signal(params.n_d_f_total_wid)
            m.d.comb += d.eq((d_shift_out << (1 + params.expanded_width))
                             >> state.d.width)

            # normalize so 1 <= n < 2
            n_leading_zeros = CLZ(2 * params.io_width)
            m.submodules.n_leading_zeros = n_leading_zeros
            m.d.comb += n_leading_zeros.sig_in.eq(state.n)
            signed_zero = Const(0, signed(1))  # force subtraction to be signed
            n_shift_s_v = (params.io_width + signed_zero + d_leading_zeros.lz
                           - n_leading_zeros.lz)
            n_shift_s = Signal.like(n_shift_s_v)
            n_shift_n_lz_out = Signal.like(state.n)
            n_shift_d_lz_out = Signal.like(state.n << d_leading_zeros.lz)
            m.d.comb += [
                n_shift_s.eq(n_shift_s_v),
                n_shift_d_lz_out.eq(state.n << d_leading_zeros.lz),
                n_shift_n_lz_out.eq(state.n << n_leading_zeros.lz),
            ]
            state.n_shift = Signal(d_leading_zeros.lz.width)
            n = Signal(params.n_d_f_total_wid)
            with m.If(n_shift_s < 0):
                m.d.comb += [
                    state.n_shift.eq(0),
                    n.eq((n_shift_d_lz_out << (1 + params.expanded_width))
                         >> state.d.width),
                ]
            with m.Else():
                m.d.comb += [
                    state.n_shift.eq(n_shift_s),
                    n.eq((n_shift_n_lz_out << (1 + params.expanded_width))
                         >> state.n.width),
                ]
            state.n = n
            state.d = d
        elif self == GoldschmidtDivOp.FEqTableLookup:
            assert state.d.width == params.n_d_f_total_wid, "invalid d width"
            # compute initial f by table lookup

            # extra bit for table entries == 1.0
            table_width = 1 + params.table_data_bits
            table = Memory(width=table_width, depth=len(params.table),
                           init=[i.bits for i in params.table])
            addr = state.d[:-params.n_d_f_int_wid][-params.table_addr_bits:]
            if sync_rom:
                table_read = table.read_port()
                m.d.comb += table_read.addr.eq(addr)
                state.insert_pipeline_register()
            else:
                table_read = table.read_port(domain="comb")
                m.d.comb += table_read.addr.eq(addr)
            m.submodules.table_read = table_read
            state.f = Signal(params.n_d_f_int_wid + params.expanded_width)
            data_shift = params.expanded_width - params.table_data_bits
            m.d.comb += state.f.eq(table_read.data << data_shift)
        elif self == GoldschmidtDivOp.MulNByF:
            assert state.n.width == params.n_d_f_total_wid, "invalid n width"
            assert state.f is not None
            assert state.f.width == params.n_d_f_total_wid, "invalid f width"
            n = Signal.like(state.n)
            m.d.comb += n.eq((state.n * state.f) >> params.expanded_width)
            state.n = n
        elif self == GoldschmidtDivOp.MulDByF:
            assert state.d.width == params.n_d_f_total_wid, "invalid d width"
            assert state.f is not None
            assert state.f.width == params.n_d_f_total_wid, "invalid f width"
            d = Signal.like(state.d)
            d_times_f = Signal.like(state.d * state.f)
            m.d.comb += [
                d_times_f.eq(state.d * state.f),
                # round the multiplication up
                d.eq((d_times_f >> params.expanded_width)
                     + (d_times_f[:params.expanded_width] != 0)),
            ]
            state.d = d
        elif self == GoldschmidtDivOp.FEq2MinusD:
            assert state.d.width == params.n_d_f_total_wid, "invalid d width"
            f = Signal.like(state.d)
            m.d.comb += f.eq((2 << params.expanded_width) - state.d)
            state.f = f
        elif self == GoldschmidtDivOp.CalcResult:
            assert state.n.width == params.n_d_f_total_wid, "invalid n width"
            assert state.n_shift is not None
            # scale to correct value
            n = state.n * (1 << state.n_shift)
            q_approx = Signal(params.io_width)
            # extra bit for if it's bigger than orig_d
            r_approx = Signal(params.io_width + 1)
            adjusted_r = Signal(signed(1 + params.io_width))
            m.d.comb += [
                q_approx.eq((state.n << state.n_shift)
                            >> params.expanded_width),
                r_approx.eq(state.orig_n - q_approx * state.orig_d),
                adjusted_r.eq(r_approx - state.orig_d),
            ]
            state.quotient = Signal(params.io_width)
            state.remainder = Signal(params.io_width)

            with m.If(adjusted_r >= 0):
                m.d.comb += [
                    state.quotient.eq(q_approx + 1),
                    state.remainder.eq(adjusted_r),
                ]
            with m.Else():
                m.d.comb += [
                    state.quotient.eq(q_approx),
                    state.remainder.eq(r_approx),
                ]
        else:
            assert False, f"unimplemented GoldschmidtDivOp: {self}"


@plain_data(repr=False)
class GoldschmidtDivState:
    __slots__ = ("orig_n", "orig_d", "n", "d",
                 "f", "quotient", "remainder", "n_shift")

    def __init__(self, orig_n, orig_d, n, d,
                 f=None, quotient=None, remainder=None, n_shift=None):
        assert isinstance(orig_n, int)
        assert isinstance(orig_d, int)
        assert isinstance(n, FixedPoint)
        assert isinstance(d, FixedPoint)
        assert f is None or isinstance(f, FixedPoint)
        assert quotient is None or isinstance(quotient, int)
        assert remainder is None or isinstance(remainder, int)
        assert n_shift is None or isinstance(n_shift, int)
        self.orig_n = orig_n
        """original numerator"""

        self.orig_d = orig_d
        """original denominator"""

        self.n = n
        """numerator -- N_prime[i] in the paper's algorithm 2"""

        self.d = d
        """denominator -- D_prime[i] in the paper's algorithm 2"""

        self.f = f
        """current factor -- F_prime[i] in the paper's algorithm 2"""

        self.quotient = quotient
        """final quotient"""

        self.remainder = remainder
        """final remainder"""

        self.n_shift = n_shift
        """amount the numerator needs to be left-shifted at the end of the
        algorithm.
        """

    def __repr__(self):
        fields_str = []
        for field in fields(GoldschmidtDivState):
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, int) and field != "n_shift":
                fields_str.append(f"{field}={hex(value)}")
            else:
                fields_str.append(f"{field}={value!r}")
        return f"GoldschmidtDivState({', '.join(fields_str)})"


def goldschmidt_div(n, d, params, trace=lambda state: None):
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
        trace: Function[[GoldschmidtDivState], None]
            called with the initial state and the state after executing each
            operation in `params.ops`.

        returns: tuple[int, int]
            the quotient and remainder. a tuple of two `width`-bit unsigned
            integers.
    """
    assert isinstance(params, GoldschmidtDivParams)
    assert isinstance(d, int) and 0 < d < (1 << params.io_width)
    assert isinstance(n, int) and 0 <= n < (d << params.io_width)

    # this whole algorithm is done with fixed-point arithmetic where values
    # have `width` fractional bits

    state = GoldschmidtDivState(
        orig_n=n,
        orig_d=d,
        n=FixedPoint(n, params.io_width),
        d=FixedPoint(d, params.io_width),
    )

    trace(state)
    for op in params.ops:
        op.run(params, state)
        trace(state)

    assert state.quotient is not None
    assert state.remainder is not None

    return state.quotient, state.remainder


@plain_data(eq=False)
class GoldschmidtDivHDLState:
    __slots__ = ("m", "orig_n", "orig_d", "n", "d",
                 "f", "quotient", "remainder", "n_shift")

    __signal_name_prefix = "state_"

    def __init__(self, m, orig_n, orig_d, n, d,
                 f=None, quotient=None, remainder=None, n_shift=None):
        assert isinstance(m, Module)
        assert isinstance(orig_n, Signal)
        assert isinstance(orig_d, Signal)
        assert isinstance(n, Signal)
        assert isinstance(d, Signal)
        assert f is None or isinstance(f, Signal)
        assert quotient is None or isinstance(quotient, Signal)
        assert remainder is None or isinstance(remainder, Signal)
        assert n_shift is None or isinstance(n_shift, Signal)

        self.m = m
        """The HDL Module"""

        self.orig_n = orig_n
        """original numerator"""

        self.orig_d = orig_d
        """original denominator"""

        self.n = n
        """numerator -- N_prime[i] in the paper's algorithm 2"""

        self.d = d
        """denominator -- D_prime[i] in the paper's algorithm 2"""

        self.f = f
        """current factor -- F_prime[i] in the paper's algorithm 2"""

        self.quotient = quotient
        """final quotient"""

        self.remainder = remainder
        """final remainder"""

        self.n_shift = n_shift
        """amount the numerator needs to be left-shifted at the end of the
        algorithm.
        """

        # old_signals must be set last
        self.old_signals = defaultdict(list)

    def __setattr__(self, name, value):
        assert isinstance(name, str)
        if name.startswith("_"):
            return super().__setattr__(name, value)
        try:
            old_signals = self.old_signals[name]
        except AttributeError:
            # haven't yet finished __post_init__
            return super().__setattr__(name, value)
        assert name != "m" and name != "old_signals", f"can't write to {name}"
        assert isinstance(value, Signal)
        value.name = f"{self.__signal_name_prefix}{name}_{len(old_signals)}"
        old_signal = getattr(self, name, None)
        if old_signal is not None:
            assert isinstance(old_signal, Signal)
            old_signals.append(old_signal)
        return super().__setattr__(name, value)

    def insert_pipeline_register(self):
        old_prefix = self.__signal_name_prefix
        try:
            for field in fields(GoldschmidtDivHDLState):
                if field.startswith("_") or field == "m":
                    continue
                old_sig = getattr(self, field, None)
                if old_sig is None:
                    continue
                assert isinstance(old_sig, Signal)
                new_sig = Signal.like(old_sig)
                setattr(self, field, new_sig)
                self.m.d.sync += new_sig.eq(old_sig)
        finally:
            self.__signal_name_prefix = old_prefix


class GoldschmidtDivHDL(Elaboratable):
    """ Goldschmidt division algorithm.

        based on:
        Even, G., Seidel, P. M., & Ferguson, W. E. (2003).
        A Parametric Error Analysis of Goldschmidt's Division Algorithm.
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.1238&rep=rep1&type=pdf

        attributes:
        params: GoldschmidtDivParams
            the goldschmidt division algorithm parameters.
        pipe_reg_indexes: list[int]
            the operation indexes where pipeline registers should be inserted.
            duplicate values mean multiple registers should be inserted for
            that operation index -- this is useful to allow yosys to spread a
            multiplication across those multiple pipeline stages.
        sync_rom: bool
            true if the rom should be read synchronously rather than
            combinatorially, incurring an extra clock cycle of latency.
        n: Signal(unsigned(2 * params.io_width))
            input numerator. a `2 * params.io_width`-bit unsigned integer.
            must be less than `d << params.io_width`, otherwise the quotient
            wouldn't fit in `params.io_width` bits.
        d: Signal(unsigned(params.io_width))
            input denominator. a `params.io_width`-bit unsigned integer.
            must not be zero.
        q: Signal(unsigned(params.io_width))
            output quotient. only valid when `n < (d << params.io_width)`.
        r: Signal(unsigned(params.io_width))
            output remainder. only valid when `n < (d << params.io_width)`.
        trace: list[GoldschmidtDivHDLState]
            list of the initial state and the state after executing each
            operation in `params.ops`.
    """

    @property
    def total_pipeline_registers(self):
        """the total number of pipeline registers"""
        return len(self.pipe_reg_indexes) + self.sync_rom

    def __init__(self, params, pipe_reg_indexes=(), sync_rom=False):
        assert isinstance(params, GoldschmidtDivParams)
        assert isinstance(sync_rom, bool)
        self.params = params
        self.pipe_reg_indexes = sorted(int(i) for i in pipe_reg_indexes)
        self.sync_rom = sync_rom
        self.n = Signal(unsigned(2 * params.io_width))
        self.d = Signal(unsigned(params.io_width))
        self.q = Signal(unsigned(params.io_width))
        self.r = Signal(unsigned(params.io_width))

        # in constructor so we get trace without needing to call elaborate
        state = GoldschmidtDivHDLState(
            m=Module(),
            orig_n=self.n,
            orig_d=self.d,
            n=self.n,
            d=self.d)

        self.trace = [replace(state)]

        # copy and reverse
        pipe_reg_indexes = list(reversed(self.pipe_reg_indexes))

        for op_index, op in enumerate(self.params.ops):
            while len(pipe_reg_indexes) > 0 \
                    and pipe_reg_indexes[-1] <= op_index:
                pipe_reg_indexes.pop()
                state.insert_pipeline_register()
            op.gen_hdl(self.params, state, self.sync_rom)
            self.trace.append(replace(state))

        while len(pipe_reg_indexes) > 0:
            pipe_reg_indexes.pop()
            state.insert_pipeline_register()

        state.m.d.comb += [
            self.q.eq(state.quotient),
            self.r.eq(state.remainder),
        ]

    def elaborate(self, platform):
        return self.trace[0].m


GOLDSCHMIDT_SQRT_RSQRT_TABLE_ADDR_INT_WID = 2


@lru_cache()
def goldschmidt_sqrt_rsqrt_table(table_addr_bits, table_data_bits):
    """Generate the look-up table needed for Goldschmidt's square-root and
    reciprocal-square-root algorithm.

    arguments:
    table_addr_bits: int
        the number of address bits for the look-up table.
    table_data_bits: int
        the number of data bits for the look-up table.
    """
    assert isinstance(table_addr_bits, int) and \
        table_addr_bits >= GOLDSCHMIDT_SQRT_RSQRT_TABLE_ADDR_INT_WID
    assert isinstance(table_data_bits, int) and table_data_bits >= 1
    table = []
    table_len = 1 << table_addr_bits
    for addr in range(table_len):
        if addr == 0:
            value = FixedPoint(0, table_data_bits)
        elif (addr << 2) < table_len:
            value = None  # table entries should be unused
        else:
            table_addr_frac_wid = table_addr_bits
            table_addr_frac_wid -= GOLDSCHMIDT_SQRT_RSQRT_TABLE_ADDR_INT_WID
            max_input_value = FixedPoint(addr + 1, table_addr_bits - 2)
            max_frac_wid = max(max_input_value.frac_wid, table_data_bits)
            value = max_input_value.to_frac_wid(max_frac_wid)
            value = value.rsqrt(RoundDir.DOWN)
            value = value.to_frac_wid(table_data_bits, RoundDir.DOWN)
        table.append(value)

    # tuple for immutability
    return tuple(table)

# FIXME: add code to calculate error bounds and check that the algorithm will
# actually work (like in the goldschmidt division algorithm).
# FIXME: add code to calculate a good set of parameters based on the error
# bounds checking.


def goldschmidt_sqrt_rsqrt(radicand, io_width, frac_wid, extra_precision,
                           table_addr_bits, table_data_bits, iter_count):
    """Goldschmidt's square-root and reciprocal-square-root algorithm.

    uses algorithm based on second method at:
    https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Goldschmidt%E2%80%99s_algorithm

    arguments:
    radicand: FixedPoint(frac_wid=frac_wid)
        the input value to take the square-root and reciprocal-square-root of.
    io_width: int
        the number of bits in the input (`radicand`) and output values.
    frac_wid: int
        the number of fraction bits in the input (`radicand`) and output
        values.
    extra_precision: int
        the number of bits of internal extra precision.
    table_addr_bits: int
        the number of address bits for the look-up table.
    table_data_bits: int
        the number of data bits for the look-up table.

    returns: tuple[FixedPoint, FixedPoint]
        the square-root and reciprocal-square-root, rounded down to the
        nearest representable value. If `radicand == 0`, then the
        reciprocal-square-root value returned is zero.
    """
    assert (isinstance(radicand, FixedPoint)
            and radicand.frac_wid == frac_wid
            and 0 <= radicand.bits < (1 << io_width))
    assert isinstance(io_width, int) and io_width >= 1
    assert isinstance(frac_wid, int) and 0 <= frac_wid < io_width
    assert isinstance(extra_precision, int) and extra_precision >= io_width
    assert isinstance(table_addr_bits, int) and table_addr_bits >= 1
    assert isinstance(table_data_bits, int) and table_data_bits >= 1
    assert isinstance(iter_count, int) and iter_count >= 0
    expanded_frac_wid = frac_wid + extra_precision
    s = radicand.to_frac_wid(expanded_frac_wid)
    sqrt_rshift = extra_precision
    rsqrt_rshift = extra_precision
    while s != 0 and s < 1:
        s = (s * 4).to_frac_wid(expanded_frac_wid)
        sqrt_rshift += 1
        rsqrt_rshift -= 1
    while s >= 4:
        s = s.div(4, expanded_frac_wid)
        sqrt_rshift -= 1
        rsqrt_rshift += 1
    table = goldschmidt_sqrt_rsqrt_table(table_addr_bits=table_addr_bits,
                                         table_data_bits=table_data_bits)
    # core goldschmidt sqrt/rsqrt algorithm:
    # initial setup:
    table_addr_frac_wid = table_addr_bits
    table_addr_frac_wid -= GOLDSCHMIDT_SQRT_RSQRT_TABLE_ADDR_INT_WID
    addr = s.to_frac_wid(table_addr_frac_wid, RoundDir.DOWN)
    assert 0 <= addr.bits < (1 << table_addr_bits), "table addr out of range"
    f = table[addr.bits]
    assert f is not None, "accessed invalid table entry"
    # use with_frac_wid to fix IDE type deduction
    f = FixedPoint.with_frac_wid(f, expanded_frac_wid, RoundDir.DOWN)
    x = (s * f).to_frac_wid(expanded_frac_wid, RoundDir.DOWN)
    h = (f * 0.5).to_frac_wid(expanded_frac_wid, RoundDir.DOWN)
    for _ in range(iter_count):
        # iteration step:
        f = (1.5 - x * h).to_frac_wid(expanded_frac_wid, RoundDir.DOWN)
        x = (x * f).to_frac_wid(expanded_frac_wid, RoundDir.DOWN)
        h = (h * f).to_frac_wid(expanded_frac_wid, RoundDir.DOWN)
    r = 2 * h
    # now `x` is approximately `sqrt(s)` and `r` is approximately `rsqrt(s)`

    sqrt = FixedPoint(x.bits >> sqrt_rshift, frac_wid)
    rsqrt = FixedPoint(r.bits >> rsqrt_rshift, frac_wid)

    next_sqrt = FixedPoint(sqrt.bits + 1, frac_wid)
    if next_sqrt * next_sqrt <= radicand:
        sqrt = next_sqrt

    next_rsqrt = FixedPoint(rsqrt.bits + 1, frac_wid)
    if next_rsqrt * next_rsqrt * radicand <= 1 and radicand != 0:
        rsqrt = next_rsqrt
    return sqrt, rsqrt
