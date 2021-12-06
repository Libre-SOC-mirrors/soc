# based on microwatt plru.vhdl

from nmigen import Elaboratable, Signal, Array, Module, Mux, Const, Cat
from nmigen.cli import rtlil
from nmigen.lib.coding import Decoder


class PLRU(Elaboratable):

    def __init__(self, BITS=2):
        self.BITS = BITS
        self.acc_i = Signal(BITS)
        self.acc_en = Signal()
        self.lru_o = Signal(BITS)

    def elaborate(self, platform):
        m = Module()
        comb, sync = m.d.comb, m.d.sync

        tree = Array(Signal(name="tree%d" % i) for i in range(self.BITS))

        # XXX Check if we can turn that into a little ROM instead that
        # takes the tree bit vector and returns the LRU. See if it's better
        # in term of FPGA resouces usage...
        node = Const(0, self.BITS)
        for i in range(self.BITS):
            # report "GET: i:" & integer'image(i) & " node:" &
            # integer'image(node) & " val:" & Signal()'image(tree(node))
            comb += self.lru_o[self.BITS-1-i].eq(tree[node])
            if i != self.BITS-1:
                node_next = Signal(self.BITS)
                node2 = Signal(self.BITS)
                comb += node2.eq(node << 1)
                comb += node_next.eq(Mux(tree[node2], node2+2, node2+1))
                node = node_next

        with m.If(self.acc_en):
            node = Const(0, self.BITS)
            for i in range(self.BITS):
                # report "GET: i:" & integer'image(i) & " node:" &
                # integer'image(node) & " val:" & Signal()'image(tree(node))
                abit = self.acc_i[self.BITS-1-i]
                sync += tree[node].eq(~abit)
                if i != self.BITS-1:
                    node_next = Signal(self.BITS)
                    node2 = Signal(self.BITS)
                    comb += node2.eq(node << 1)
                    comb += node_next.eq(Mux(abit, node2+2, node2+1))
                    node = node_next

        return m

    def ports(self):
        return [self.acc_en, self.lru_o, self.acc_i]


class PLRUs(Elaboratable):
    def __init__(self, n_plrus, n_bits, out=None):
        self.n_plrus = n_plrus
        self.n_bits = n_bits
        self.valid = Signal()
        self.way = Signal(n_bits)
        self.index = Signal(n_plrus.bit_length())
        if out is None:
            out = Array(Signal(n_bits, name="plru_out%d" % x) \
                             for x in range(n_plrus))
        self.out = out

    def elaborate(self, platform):
        """Generate TLB PLRUs
        """
        m = Module()
        comb = m.d.comb

        if self.n_plrus == 0:
            return m

        # Binary-to-Unary one-hot, enabled by valid
        m.submodules.te = te = Decoder(self.n_plrus)
        comb += te.n.eq(~self.valid)
        comb += te.i.eq(self.index)

        for i in range(self.n_plrus):
            # PLRU interface
            m.submodules["plru_%d" % i] = plru = PLRU(self.n_bits)

            comb += plru.acc_en.eq(te.o[i])
            comb += plru.acc_i.eq(self.way)
            comb += self.out[i].eq(plru.lru_o)

        return m

    def ports(self):
        return [self.valid, self.way, self.index] + list(self.out)


if __name__ == '__main__':
    dut = PLRU(2)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_plru.il", "w") as f:
        f.write(vl)


    dut = PLRUs(4, 2)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_plrus.il", "w") as f:
        f.write(vl)


