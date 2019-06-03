""" Mitch Alsup 6600-style LD/ST Memory Scoreboard Matrix (sparse vector)

6600 LD/ST Dependency Table Matrix inputs / outputs
---------------------------------------------------

Relevant comments (p45-46):

* If there are no WAR dependencies on a Load instruction with a computed
  address it can assert Bank_Addressable and Translate_Addressable.

* If there are no RAW dependencies on a Store instruction with both a
  write permission and store data present it can assert Bank_Addressable

Relevant bugreports:

* http://bugs.libre-riscv.org/show_bug.cgi?id=81

Notes:

* Load Hit (or Store Hit with Data) are asserted by the LD/ST Computation
  Unit when it has data and address ready

* Asserting the ld_hit_i (or stwd_hit_i) *requires* that the output be
  captured or at least taken into consideration for the next LD/STs
  *right then*.  Failure to observe the xx_hold_xx_o *will* result in
  data corruption, as they are *only* asserted if xx_hit_i is asserted

* The hold signals still have to go through "maybe address clashes"
  detection, they cannot just be used as-is to stop a LD/ST.

"""

from nmigen.compat.sim import run_simulation
from nmigen.cli import verilog, rtlil
from nmigen import Module, Signal, Elaboratable, Array, Cat, Const

from ldst_dep_cell import LDSTDepCell


class LDSTDepMatrix(Elaboratable):
    """ implements 11.4.12 mitch alsup LD/ST Dependency Matrix, p46
        actually a sparse matrix along the diagonal.

        load-hold-store and store-hold-load accumulate in a priority-picking
        fashion, ORing together.  the OR gate from the dependency cell is
        here.
    """
    def __init__(self, n_ldst):
        self.n_ldst = n_ldst                  # X and Y (FUs)
        self.load_i = Signal(n_ldst, reset_less=True)  # load pending in
        self.stor_i = Signal(n_ldst, reset_less=True)  # store pending in
        self.issue_i = Signal(n_ldst, reset_less=True) # Issue in

        self.load_hit_i = Signal(n_ldst, reset_less=True) # load hit in
        self.stwd_hit_i = Signal(n_ldst, reset_less=True) # store w/data hit in

        # outputs
        self.ld_hold_st_o = Signal(n_ldst, reset_less=True) # load holds st out
        self.st_hold_ld_o = Signal(n_ldst, reset_less=True) # st holds load out

    def elaborate(self, platform):
        m = Module()

        # ---
        # matrix of dependency cells.  actually, LDSTDepCell is a row, now
        # ---
        dm = Array(LDSTDepCell(self.n_ldst) for f in range(self.n_ldst))
        for fu in range(self.n_ldst):
            setattr(m.submodules, "dm_fu%d" % (fu), dm[fu])

        # ---
        # connect Function Unit vector, all horizontal
        # ---
        lhs_l = []
        shl_l = []
        load_l = []
        stor_l = []
        issue_l = []
        lh_l = []
        sh_l = []
        for fu in range(self.n_ldst):
            dc = dm[fu]
            # accumulate load-hold-store / store-hold-load bits (horizontal)
            lhs_l.append(dc.ld_hold_st_o)
            shl_l.append(dc.st_hold_ld_o)
            # accumulate inputs (for Cat'ing later) - TODO: must be a better way
            load_l.append(dc.load_h_i)
            stor_l.append(dc.stor_h_i)
            issue_l.append(dc.issue_i)

            # load-hit and store-with-data-hit go in vertically (top)
            m.d.comb += [dc.load_hit_i.eq(self.load_hit_i),
                         dc.stwd_hit_i.eq(self.stwd_hit_i)
                        ]

        # connect cell inputs using Cat(*list_of_stuff)
        m.d.comb += [Cat(*load_l).eq(self.load_i),
                     Cat(*stor_l).eq(self.stor_i),
                     Cat(*issue_l).eq(self.issue_i),
                    ]
        # connect the load-hold-store / store-hold-load OR-accumulated outputs
        m.d.comb += self.ld_hold_st_o.eq(Cat(*lhs_l))
        m.d.comb += self.st_hold_ld_o.eq(Cat(*shl_l))

        # the load/store input also needs to be connected to "top" (vertically)
        for fu in range(self.n_ldst):
            load_v_l = []
            stor_v_l = []
            for fux in range(self.n_ldst):
                dc = dm[fux]
                load_v_l.append(dc.load_v_i[fu])
                stor_v_l.append(dc.stor_v_i[fu])
            m.d.comb += [Cat(*load_v_l).eq(self.load_i),
                         Cat(*stor_v_l).eq(self.stor_i),
                        ]

        return m

    def __iter__(self):
        yield self.load_i
        yield self.stor_i
        yield self.issue_i
        yield self.load_hit_i
        yield self.stwd_hit_i
        yield self.ld_hold_st_o
        yield self.st_hold_ld_o

    def ports(self):
        return list(self)

def d_matrix_sim(dut):
    """ XXX TODO
    """
    yield dut.dest_i.eq(1)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.src1_i.eq(1)
    yield dut.issue_i.eq(1)
    yield
    yield dut.issue_i.eq(0)
    yield
    yield dut.go_rd_i.eq(1)
    yield
    yield dut.go_rd_i.eq(0)
    yield
    yield dut.go_wr_i.eq(1)
    yield
    yield dut.go_wr_i.eq(0)
    yield

def test_d_matrix():
    dut = LDSTDepMatrix(n_ldst=4)
    vl = rtlil.convert(dut, ports=dut.ports())
    with open("test_ld_st_matrix.il", "w") as f:
        f.write(vl)

    run_simulation(dut, d_matrix_sim(dut), vcd_name='test_ld_st_matrix.vcd')

if __name__ == '__main__':
    test_d_matrix()
