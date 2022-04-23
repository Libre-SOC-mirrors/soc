# SPDX-License-Identifier: LGPL-3-or-later
# Copyright 2022 Jacob Lifshay programmerjake@gmail.com

# Funded by NLnet Assure Programme 2021-02-052, https://nlnet.nl/assure part
# of Horizon 2020 EU Programme 957073.

from dataclasses import dataclass, field
import math
import enum
from fractions import Fraction


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
            # compute number of bits that should be removed from value
            del_bits = value.frac_wid - frac_wid
            if del_bits == 0:
                return value
            if del_bits < 0:  # add bits
                return FixedPoint(value.bits << -del_bits,
                                  frac_wid)
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
    n: FixedPoint
    """numerator -- N_prime[i] in the paper's algorithm 2"""
    d: FixedPoint
    """denominator -- D_prime[i] in the paper's algorithm 2"""
    f: "FixedPoint | None" = None
    """current factor -- F_prime[i] in the paper's algorithm 2"""
    result: "int | None" = None
    """final result"""
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
    # tuple to be immutable
    table: "tuple[FixedPoint, ...]" = field(init=False)
    """the lookup-table"""
    ops: "tuple[GoldschmidtDivOp, ...]" = field(init=False)
    """the operations needed to perform the goldschmidt division algorithm."""

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
        assert self.io_width >= self.table_addr_bits
        min_numerator = (1 << self.table_addr_bits) + addr
        denominator = 1 << self.table_addr_bits
        values_per_table_entry = 1 << (self.io_width - self.table_addr_bits)
        max_numerator = min_numerator + values_per_table_entry
        min_input = Fraction(min_numerator, denominator)
        max_input = Fraction(max_numerator, denominator)
        return min_input, max_input

    def table_value_exact_range(self, addr):
        """return the range of values as `Fraction`s used for the table entry
        with address `addr`."""
        min_value, max_value = self.table_input_exact_range(addr)
        # division swaps min/max
        return 1 / max_value, 1 / min_value

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
        table = []
        for addr in range(1 << self.table_addr_bits):
            table.append(FixedPoint.with_frac_wid(self.table_exact_value(addr),
                                                  self.table_data_bits,
                                                  RoundDir.DOWN))
        # we have to use object.__setattr__ since frozen=True
        object.__setattr__(self, "table", tuple(table))
        object.__setattr__(self, "ops", tuple(_goldschmidt_div_ops(self)))

    @staticmethod
    def get(io_width):
        """ find efficient parameters for a goldschmidt division algorithm
        with `params.io_width == io_width`.
        """
        assert isinstance(io_width, int) and io_width >= 1
        for extra_precision in range(io_width * 2):
            for table_addr_bits in range(3, 7 + 1):
                table_data_bits = io_width + extra_precision
                try:
                    return GoldschmidtDivParams(
                        io_width=io_width,
                        extra_precision=extra_precision,
                        table_addr_bits=table_addr_bits,
                        table_data_bits=table_data_bits)
                except ParamsNotAccurateEnough:
                    pass
        raise ValueError(f"can't find working parameters for a goldschmidt "
                         f"division algorithm with io_width={io_width}")

    @property
    def expanded_width(self):
        """the total number of bits of precision used inside the algorithm."""
        return self.io_width + self.extra_precision


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

            # avoid incorrectly rounding down
            n = n.to_frac_wid(params.io_width, round_dir=RoundDir.UP)
            state.result = math.floor(n)
        else:
            assert False, f"unimplemented GoldschmidtDivOp: {self}"


def _goldschmidt_div_ops(params):
    """ Goldschmidt division algorithm.

        based on:
        Even, G., Seidel, P. M., & Ferguson, W. E. (2003).
        A Parametric Error Analysis of Goldschmidt's Division Algorithm.
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.1238&rep=rep1&type=pdf

        arguments:
        params: GoldschmidtDivParams
            the parameters for the algorithm

        yields: GoldschmidtDivOp
            the operations needed to perform the division.
    """
    assert isinstance(params, GoldschmidtDivParams)

    # establish assumptions of the paper's error analysis (section 3.1):

    # 1. normalize so A (numerator) and B (denominator) are in [1, 2)
    yield GoldschmidtDivOp.Normalize

    # 2. ensure all relative errors from directed rounding are <= 1 / 4.
    # the assumption is met by multipliers with > 4-bits precision
    _assert_accuracy(params.expanded_width > 4)

    # 3. require `abs(e[0]) + 3 * d[0] / 2 + f[0] < 1 / 2`.

    # maximum `abs(e[0])`
    max_abs_e0 = 0
    # maximum `d[0]`
    max_d0 = 0
    # `f[i] = 0` for all `i`
    fi = 0
    for addr in range(params.table_addr_count):
        # `F_prime[-1] = (1 - e[0]) / B`
        # => `e[0] = 1 - B * F_prime[-1]`
        min_b, max_b = params.table_input_exact_range(addr)
        f_prime_m1 = params.table[addr].as_fraction()
        assert min_b >= 0 and f_prime_m1 >= 0, \
            "only positive quadrant of interval multiplication implemented"
        min_product = min_b * f_prime_m1
        max_product = max_b * f_prime_m1
        # negation swaps min/max
        min_e0 = 1 - max_product
        max_e0 = 1 - min_product
        max_abs_e0 = max(max_abs_e0, abs(min_e0), abs(max_e0))

        # `D_prime[0] = (1 + d[0]) * B * F_prime[-1]`
        # `D_prime[0] = abs_round_err + B * F_prime[-1]`
        # => `d[0] = abs_round_err / (B * F_prime[-1])`
        max_abs_round_err = Fraction(1, 1 << params.expanded_width)
        assert min_product > 0 and max_abs_round_err >= 0, \
            "only positive quadrant of interval division implemented"
        # division swaps divisor's min/max
        max_d0 = max(max_d0, max_abs_round_err / min_product)

    _assert_accuracy(max_abs_e0 + 3 * max_d0 / 2 + fi < Fraction(1, 2))

    # 4. the initial approximation F'[-1] of 1/B is in [1/2, 1].
    # (B is the denominator)

    for addr in range(params.table_addr_count):
        f_prime_m1 = params.table[addr]
        _assert_accuracy(0.5 <= f_prime_m1 <= 1)

    yield GoldschmidtDivOp.FEqTableLookup

    # we use Setting I (section 4.1 of the paper)

    min_bits_of_precision = 1
    # FIXME: calculate error and check if it's small enough
    while min_bits_of_precision < params.io_width * 2:
        yield GoldschmidtDivOp.MulNByF
        yield GoldschmidtDivOp.MulDByF
        yield GoldschmidtDivOp.FEq2MinusD

        min_bits_of_precision *= 2

    yield GoldschmidtDivOp.CalcResult


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

        returns: int
            the quotient. a `width`-bit unsigned integer.
    """
    assert isinstance(params, GoldschmidtDivParams)
    assert isinstance(d, int) and 0 < d < (1 << params.io_width)
    assert isinstance(n, int) and 0 <= n < (d << params.io_width)

    # this whole algorithm is done with fixed-point arithmetic where values
    # have `width` fractional bits

    state = GoldschmidtDivState(
        n=FixedPoint(n, params.io_width),
        d=FixedPoint(d, params.io_width),
    )

    for op in params.ops:
        op.run(params, state)

    assert state.result is not None

    return state.result
