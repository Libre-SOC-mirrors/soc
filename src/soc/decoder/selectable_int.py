import unittest
from copy import copy
from soc.decoder.power_fields import BitRange
from operator import (add, sub, mul, floordiv, truediv, mod, or_, and_, xor,
                      neg, inv, lshift, rshift)


def check_extsign(a, b):
    if isinstance(b, FieldSelectableInt):
        b = b.get_range()
    if isinstance(b, int):
        return SelectableInt(b, a.bits)
    if b.bits != 256:
        return b
    return SelectableInt(b.value, a.bits)


class FieldSelectableInt:
    """FieldSelectableInt: allows bit-range selection onto another target
    """

    def __init__(self, si, br):
        self.si = si  # target selectable int
        if isinstance(br, list) or isinstance(br, tuple):
            _br = BitRange()
            for i, v in enumerate(br):
                _br[i] = v
            br = _br
        self.br = br  # map of indices.

    def eq(self, b):
        if isinstance(b, SelectableInt):
            for i in range(b.bits):
                self[i] = b[i]
        else:
            self.si = copy(b.si)
            self.br = copy(b.br)

    def _op(self, op, b):
        vi = self.get_range()
        vi = op(vi, b)
        return self.merge(vi)

    def _op1(self, op):
        vi = self.get_range()
        vi = op(vi)
        return self.merge(vi)

    def __getitem__(self, key):
        print("getitem", key, self.br)
        if isinstance(key, SelectableInt):
            key = key.value
        if isinstance(key, int):
            key = self.br[key]  # don't do POWER 1.3.4 bit-inversion
            return self.si[key]
        if isinstance(key, slice):
            key = self.br[key]
            return selectconcat(*[self.si[x] for x in key])

    def __setitem__(self, key, value):
        if isinstance(key, SelectableInt):
            key = key.value
        key = self.br[key]  # don't do POWER 1.3.4 bit-inversion
        if isinstance(key, int):
            return self.si.__setitem__(key, value)
        else:
            if not isinstance(value, SelectableInt):
                value = SelectableInt(value, bits=len(key))
            for i, k in enumerate(key):
                self.si[k] = value[i]

    def __negate__(self):
        return self._op1(negate)

    def __invert__(self):
        return self._op1(inv)

    def __add__(self, b):
        return self._op(add, b)

    def __sub__(self, b):
        return self._op(sub, b)

    def __mul__(self, b):
        return self._op(mul, b)

    def __div__(self, b):
        return self._op(truediv, b)

    def __mod__(self, b):
        return self._op(mod, b)

    def __and__(self, b):
        return self._op(and_, b)

    def __or__(self, b):
        return self._op(or_, b)

    def __xor__(self, b):
        return self._op(xor, b)

    def get_range(self):
        vi = SelectableInt(0, len(self.br))
        for k, v in self.br.items():
            vi[k] = self.si[v]
        return vi

    def merge(self, vi):
        fi = copy(self)
        for i, v in fi.br.items():
            fi.si[v] = vi[i]
        return fi

    def __repr__(self):
        return "FieldSelectableInt(si=%s, br=%s)" % (self.si, self.br)

    def asint(self, msb0=False):
        res = 0
        brlen = len(self.br)
        for i, key in self.br.items():
            bit = self.si[key].value
            #print("asint", i, key, bit)
            res |= bit << ((brlen-i-1) if msb0 else i)
        return res


class FieldSelectableIntTestCase(unittest.TestCase):
    def test_arith(self):
        a = SelectableInt(0b10101, 5)
        b = SelectableInt(0b011, 3)
        br = BitRange()
        br[0] = 0
        br[1] = 2
        br[2] = 3
        fs = FieldSelectableInt(a, br)
        c = fs + b
        print(c)
        #self.assertEqual(c.value, a.value + b.value)

    def test_select(self):
        a = SelectableInt(0b00001111, 8)
        br = BitRange()
        br[0] = 0
        br[1] = 1
        br[2] = 4
        br[3] = 5
        fs = FieldSelectableInt(a, br)

        self.assertEqual(fs.get_range(), 0b0011)

    def test_select_range(self):
        a = SelectableInt(0b00001111, 8)
        br = BitRange()
        br[0] = 0
        br[1] = 1
        br[2] = 4
        br[3] = 5
        fs = FieldSelectableInt(a, br)

        self.assertEqual(fs[2:4], 0b11)

        fs[0:2] = 0b10
        self.assertEqual(fs.get_range(), 0b1011)

class SelectableInt:
    """SelectableInt - a class that behaves exactly like python int

    this class is designed to mirror precisely the behaviour of python int.
    the only difference is that it must contain the context of the bitwidth
    (number of bits) associated with that integer.

    FieldSelectableInt can then operate on partial bits, and because there
    is a bit width associated with SelectableInt, slices operate correctly
    including negative start/end points.
    """

    def __init__(self, value, bits):
        if isinstance(value, SelectableInt):
            value = value.value
        mask = (1 << bits) - 1
        self.value = value & mask
        self.bits = bits
        self.overflow = (value & ~mask) != 0

    def eq(self, b):
        self.value = b.value
        self.bits = b.bits

    def to_signed_int(self):
        print ("to signed?", self.value & (1<<(self.bits-1)), self.value)
        if self.value & (1<<(self.bits-1)) != 0: # negative
            res = self.value - (1<<self.bits)
            print ("    val -ve:", self.bits, res)
        else:
            res = self.value
            print ("    val +ve:", res)
        return res

    def _op(self, op, b):
        if isinstance(b, int):
            b = SelectableInt(b, self.bits)
        b = check_extsign(self, b)
        assert b.bits == self.bits
        return SelectableInt(op(self.value, b.value), self.bits)

    def __add__(self, b):
        return self._op(add, b)

    def __sub__(self, b):
        return self._op(sub, b)

    def __mul__(self, b):
        # different case: mul result needs to fit the total bitsize
        if isinstance(b, int):
            b = SelectableInt(b, self.bits)
        print("SelectableInt mul", hex(self.value), hex(b.value),
              self.bits, b.bits)
        return SelectableInt(self.value * b.value, self.bits + b.bits)

    def __floordiv__(self, b):
        return self._op(floordiv, b)

    def __truediv__(self, b):
        return self._op(truediv, b)

    def __mod__(self, b):
        return self._op(mod, b)

    def __and__(self, b):
        return self._op(and_, b)

    def __or__(self, b):
        return self._op(or_, b)

    def __xor__(self, b):
        return self._op(xor, b)

    def __abs__(self):
        print("abs", self.value & (1 << (self.bits-1)))
        if self.value & (1 << (self.bits-1)) != 0:
            return -self
        return self

    def __rsub__(self, b):
        if isinstance(b, int):
            b = SelectableInt(b, self.bits)
        b = check_extsign(self, b)
        assert b.bits == self.bits
        return SelectableInt(b.value - self.value, self.bits)

    def __radd__(self, b):
        if isinstance(b, int):
            b = SelectableInt(b, self.bits)
        b = check_extsign(self, b)
        assert b.bits == self.bits
        return SelectableInt(b.value + self.value, self.bits)

    def __rxor__(self, b):
        b = check_extsign(self, b)
        assert b.bits == self.bits
        return SelectableInt(self.value ^ b.value, self.bits)

    def __invert__(self):
        return SelectableInt(~self.value, self.bits)

    def __neg__(self):
        res = SelectableInt((~self.value) + 1, self.bits)
        print ("neg", hex(self.value), hex(res.value))
        return res

    def __lshift__(self, b):
        b = check_extsign(self, b)
        return SelectableInt(self.value << b.value, self.bits)

    def __rshift__(self, b):
        b = check_extsign(self, b)
        return SelectableInt(self.value >> b.value, self.bits)

    def __getitem__(self, key):
        if isinstance(key, SelectableInt):
            key = key.value
        if isinstance(key, int):
            assert key < self.bits, "key %d accessing %d" % (key, self.bits)
            assert key >= 0
            # NOTE: POWER 3.0B annotation order!  see p4 1.3.2
            # MSB is indexed **LOWEST** (sigh)
            key = self.bits - (key + 1)

            value = (self.value >> key) & 1
            print("getitem", key, self.bits, hex(self.value), value)
            return SelectableInt(value, 1)
        elif isinstance(key, slice):
            assert key.step is None or key.step == 1
            assert key.start < key.stop
            assert key.start >= 0
            assert key.stop <= self.bits

            stop = self.bits - key.start
            start = self.bits - key.stop

            bits = stop - start
            #print ("__getitem__ slice num bits", start, stop, bits)
            mask = (1 << bits) - 1
            value = (self.value >> start) & mask
            print("getitem", stop, start, self.bits, hex(self.value), value)
            return SelectableInt(value, bits)

    def __setitem__(self, key, value):
        if isinstance(key, SelectableInt):
            key = key.value
        print("setitem", key, self.bits, hex(self.value))
        if isinstance(key, int):
            assert key < self.bits
            assert key >= 0
            key = self.bits - (key + 1)
            if isinstance(value, SelectableInt):
                assert value.bits == 1
                value = value.value

            value = value << key
            mask = 1 << key
            self.value = (self.value & ~mask) | (value & mask)
        elif isinstance(key, slice):
            assert key.step is None or key.step == 1
            assert key.start < key.stop
            assert key.start >= 0
            assert key.stop <= self.bits

            stop = self.bits - key.start
            start = self.bits - key.stop

            bits = stop - start
            #print ("__setitem__ slice num bits", bits)
            if isinstance(value, SelectableInt):
                assert value.bits == bits, "%d into %d" % (value.bits, bits)
                value = value.value
            mask = ((1 << bits) - 1) << start
            value = value << start
            self.value = (self.value & ~mask) | (value & mask)

    def __ge__(self, other):
        if isinstance(other, FieldSelectableInt):
            other = other.get_range()
        if isinstance(other, SelectableInt):
            other = check_extsign(self, other)
            assert other.bits == self.bits
            other = other.to_signed_int()
        if isinstance(other, int):
            return onebit(self.to_signed_int() >= other)
        assert False

    def __le__(self, other):
        if isinstance(other, FieldSelectableInt):
            other = other.get_range()
        if isinstance(other, SelectableInt):
            other = check_extsign(self, other)
            assert other.bits == self.bits
            other = other.to_signed_int()
        if isinstance(other, int):
            return onebit(self.to_signed_int() <= other)
        assert False

    def __gt__(self, other):
        if isinstance(other, FieldSelectableInt):
            other = other.get_range()
        if isinstance(other, SelectableInt):
            other = check_extsign(self, other)
            assert other.bits == self.bits
            other = other.to_signed_int()
        if isinstance(other, int):
            return onebit(self.to_signed_int() > other)
        assert False

    def __lt__(self, other):
        print ("SelectableInt lt", self, other)
        if isinstance(other, FieldSelectableInt):
            other = other.get_range()
        if isinstance(other, SelectableInt):
            other = check_extsign(self, other)
            assert other.bits == self.bits
            other = other.to_signed_int()
        if isinstance(other, int):
            a = self.to_signed_int()
            res = onebit(a  < other)
            print ("    a < b", a, other, res)
            return res
        assert False

    def __eq__(self, other):
        print("__eq__", self, other)
        if isinstance(other, FieldSelectableInt):
            other = other.get_range()
        if isinstance(other, SelectableInt):
            other = check_extsign(self, other)
            assert other.bits == self.bits
            other = other.value
        print ("    eq", other, self.value, other == self.value)
        if isinstance(other, int):
            return onebit(other == self.value)
        assert False

    def narrow(self, bits):
        assert bits <= self.bits
        return SelectableInt(self.value, bits)

    def __bool__(self):
        return self.value != 0

    def __repr__(self):
        return "SelectableInt(value=0x{:x}, bits={})".format(self.value,
                                                             self.bits)

    def __len__(self):
        return self.bits

    def asint(self):
        return self.value


def onebit(bit):
    return SelectableInt(1 if bit else 0, 1)


def selectltu(lhs, rhs):
    """ less-than (unsigned)
    """
    if isinstance(rhs, SelectableInt):
        rhs = rhs.value
    return onebit(lhs.value < rhs)


def selectgtu(lhs, rhs):
    """ greater-than (unsigned)
    """
    if isinstance(rhs, SelectableInt):
        rhs = rhs.value
    return onebit(lhs.value > rhs)


# XXX this probably isn't needed...
def selectassign(lhs, idx, rhs):
    if isinstance(idx, tuple):
        if len(idx) == 2:
            lower, upper = idx
            step = None
        else:
            lower, upper, step = idx
        toidx = range(lower, upper, step)
        fromidx = range(0, upper-lower, step)  # XXX eurgh...
    else:
        toidx = [idx]
        fromidx = [0]
    for t, f in zip(toidx, fromidx):
        lhs[t] = rhs[f]


def selectconcat(*args, repeat=1):
    if repeat != 1 and len(args) == 1 and isinstance(args[0], int):
        args = [SelectableInt(args[0], 1)]
    if repeat != 1:  # multiplies the incoming arguments
        tmp = []
        for i in range(repeat):
            tmp += args
        args = tmp
    res = copy(args[0])
    for i in args[1:]:
        if isinstance(i, FieldSelectableInt):
            i = i.si
        assert isinstance(i, SelectableInt), "can only concat SIs, sorry"
        res.bits += i.bits
        res.value = (res.value << i.bits) | i.value
    print("concat", repeat, res)
    return res


class SelectableIntTestCase(unittest.TestCase):
    def test_arith(self):
        a = SelectableInt(5, 8)
        b = SelectableInt(9, 8)
        c = a + b
        d = a - b
        e = a * b
        f = -a
        g = abs(f)
        h = abs(a)
        self.assertEqual(c.value, a.value + b.value)
        self.assertEqual(d.value, (a.value - b.value) & 0xFF)
        self.assertEqual(e.value, (a.value * b.value) & 0xFF)
        self.assertEqual(f.value, (-a.value) & 0xFF)
        self.assertEqual(c.bits, a.bits)
        self.assertEqual(d.bits, a.bits)
        self.assertEqual(e.bits, a.bits)
        self.assertEqual(a.bits, f.bits)
        self.assertEqual(a.bits, h.bits)

    def test_logic(self):
        a = SelectableInt(0x0F, 8)
        b = SelectableInt(0xA5, 8)
        c = a & b
        d = a | b
        e = a ^ b
        f = ~a
        self.assertEqual(c.value, a.value & b.value)
        self.assertEqual(d.value, a.value | b.value)
        self.assertEqual(e.value, a.value ^ b.value)
        self.assertEqual(f.value, 0xF0)

    def test_get(self):
        a = SelectableInt(0xa2, 8)
        # These should be big endian
        self.assertEqual(a[7], 0)
        self.assertEqual(a[0:4], 10)
        self.assertEqual(a[4:8], 2)

    def test_set(self):
        a = SelectableInt(0x5, 8)
        a[7] = SelectableInt(0, 1)
        self.assertEqual(a, 4)
        a[4:8] = 9
        self.assertEqual(a, 9)
        a[0:4] = 3
        self.assertEqual(a, 0x39)
        a[0:4] = a[4:8]
        self.assertEqual(a, 0x99)

    def test_concat(self):
        a = SelectableInt(0x1, 1)
        c = selectconcat(a, repeat=8)
        self.assertEqual(c, 0xff)
        self.assertEqual(c.bits, 8)
        a = SelectableInt(0x0, 1)
        c = selectconcat(a, repeat=8)
        self.assertEqual(c, 0x00)
        self.assertEqual(c.bits, 8)

    def test_repr(self):
        for i in range(65536):
            a = SelectableInt(i, 16)
            b = eval(repr(a))
            self.assertEqual(a, b)

    def test_cmp(self):
        a = SelectableInt(10, bits=8)
        b = SelectableInt(5, bits=8)
        self.assertTrue(a > b)
        self.assertFalse(a < b)
        self.assertTrue(a != b)
        self.assertFalse(a == b)

    def test_unsigned(self):
        a = SelectableInt(0x80, bits=8)
        b = SelectableInt(0x7f, bits=8)
        self.assertTrue(a > b)
        self.assertFalse(a < b)
        self.assertTrue(a != b)
        self.assertFalse(a == b)


if __name__ == "__main__":
    unittest.main()
