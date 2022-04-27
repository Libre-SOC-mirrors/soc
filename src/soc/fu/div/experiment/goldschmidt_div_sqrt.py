# SPDX-License-Identifier: LGPL-3-or-later
# Copyright 2022 Jacob Lifshay programmerjake@gmail.com

# Funded by NLnet Assure Programme 2021-02-052, https://nlnet.nl/assure part
# of Horizon 2020 EU Programme 957073.

from dataclasses import dataclass, field
import math
import enum
from fractions import Fraction
from types import FunctionType

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

# fix broken IDE type detection for cached_property
from typing import TYPE_CHECKING
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


@dataclass
class GoldschmidtDivState:
    orig_n: int
    """original numerator"""

    orig_d: int
    """original denominator"""

    n: FixedPoint
    """numerator -- N_prime[i] in the paper's algorithm 2"""

    d: FixedPoint
    """denominator -- D_prime[i] in the paper's algorithm 2"""

    f: "FixedPoint | None" = None
    """current factor -- F_prime[i] in the paper's algorithm 2"""

    quotient: "int | None" = None
    """final quotient"""

    remainder: "int | None" = None
    """final remainder"""

    n_shift: "int | None" = None
    """amount the numerator needs to be left-shifted at the end of the
    algorithm.
    """


class ParamsNotAccurateEnough(Exception):
    """raised when the parameters aren't accurate enough to have goldschmidt
    division work."""


def _assert_accuracy(condition, msg="not accurate enough"):
    if condition:
        return
    raise ParamsNotAccurateEnough(msg)


@dataclass(frozen=True, unsafe_hash=True)
class GoldschmidtDivParams:
    """parameters for a Goldschmidt division algorithm.
    Use `GoldschmidtDivParams.get` to find a efficient set of parameters.
    """

    io_width: int
    """bit-width of the input divisor and the result.
    the input numerator is `2 * io_width`-bits wide.
    """

    extra_precision: int
    """number of bits of additional precision used inside the algorithm."""

    table_addr_bits: int
    """the number of address bits used in the lookup-table."""

    table_data_bits: int
    """the number of data bits used in the lookup-table."""

    iter_count: int
    """the total number of iterations of the division algorithm's loop"""

    # tuple to be immutable, default so repr() works for debugging even when
    # __post_init__ hasn't finished running yet
    table: "tuple[FixedPoint, ...]" = field(init=False, default=NotImplemented)
    """the lookup-table"""

    ops: "tuple[GoldschmidtDivOp, ...]" = field(init=False,
                                                default=NotImplemented)
    """the operations needed to perform the goldschmidt division algorithm."""

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

    def __post_init__(self):
        # called by the autogenerated __init__
        assert self.io_width >= 1
        assert self.extra_precision >= 0
        assert self.table_addr_bits >= 1
        assert self.table_data_bits >= 1
        assert self.iter_count >= 1
        table = []
        for addr in range(1 << self.table_addr_bits):
            table.append(FixedPoint.with_frac_wid(self.table_exact_value(addr),
                                                  self.table_data_bits,
                                                  RoundDir.DOWN))
        # we have to use object.__setattr__ since frozen=True
        object.__setattr__(self, "table", tuple(table))
        object.__setattr__(self, "ops", tuple(self.__make_ops()))

    @property
    def expanded_width(self):
        """the total number of bits of precision used inside the algorithm."""
        return self.io_width + self.extra_precision

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
        # input numerator is `2*io_width`-bits
        max_n = (1 << (self.io_width * 2)) - 1
        max_n_shift = 0
        # normalize so 1 <= n < 2
        while max_n >= 2:
            max_n >>= 1
            max_n_shift += 1
        return max_n_shift

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
        # Require `n[i] <= n_hat` and `d[i] <= n_hat` and `f[i] = 0`
        n_hat = Fraction(0)
        for i in range(self.iter_count):
            _assert_accuracy(self.max_f(i) == 0)
            n_hat = max(n_hat, self.max_n(i), self.max_d(i))
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
        power = self._shrink_max(self.max_abs_e0 + 3 * n_hat / 2)
        for _ in range(i):
            power = self._shrink_max(power * power)

        max_rel_error = (2 * i) * n_hat + power

        min_a_over_b = Fraction(1, 2)
        max_a_over_b = Fraction(2)
        max_allowed_abs_error = max_a_over_b / (1 << self.max_n_shift)
        max_allowed_rel_error = max_allowed_abs_error / min_a_over_b

        _assert_accuracy(max_rel_error < max_allowed_rel_error,
                         f"not accurate enough: max_rel_error={max_rel_error}"
                         f" max_allowed_rel_error={max_allowed_rel_error}")

        yield GoldschmidtDivOp.CalcResult

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
        cost += 1e6 * self.iter_count
        return cost

    @staticmethod
    def get(io_width):
        """ find efficient parameters for a goldschmidt division algorithm
        with `params.io_width == io_width`.
        """
        assert isinstance(io_width, int) and io_width >= 1
        last_params = None
        last_error = None
        for extra_precision in range(io_width * 2 + 4):
            for table_addr_bits in range(1, 7 + 1):
                table_data_bits = io_width + extra_precision
                for iter_count in range(1, 2 * io_width.bit_length()):
                    try:
                        return GoldschmidtDivParams(
                            io_width=io_width,
                            extra_precision=extra_precision,
                            table_addr_bits=table_addr_bits,
                            table_data_bits=table_data_bits,
                            iter_count=iter_count)
                    except ParamsNotAccurateEnough as e:
                        last_params = (f"GoldschmidtDivParams("
                                       f"io_width={io_width!r}, "
                                       f"extra_precision={extra_precision!r}, "
                                       f"table_addr_bits={table_addr_bits!r}, "
                                       f"table_data_bits={table_data_bits!r}, "
                                       f"iter_count={iter_count!r})")
                        last_error = e
        raise ValueError(f"can't find working parameters for a goldschmidt "
                         f"division algorithm: last params: {last_params}"
                         ) from last_error


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
                state.n = (state.n * 0.5).to_frac_wid(expanded_width)
                state.n_shift += 1
        elif self == GoldschmidtDivOp.FEqTableLookup:
            # compute initial f by table lookup
            d_m_1 = state.d - 1
            d_m_1 = d_m_1.to_frac_wid(table_addr_bits, RoundDir.DOWN)
            assert 0 <= d_m_1.bits < (1 << params.table_addr_bits)
            state.f = params.table[d_m_1.bits]
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


def goldschmidt_div(n, d, params):
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

    for op in params.ops:
        op.run(params, state)

    assert state.quotient is not None
    assert state.remainder is not None

    return state.quotient, state.remainder
