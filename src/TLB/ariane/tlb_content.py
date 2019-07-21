from nmigen import Signal, Module, Cat, Const, Elaboratable

from TLB.ariane.ptw import TLBUpdate, PTE


class TLBEntry:
    def __init__(self, asid_width):
        self.asid = Signal(asid_width)
        # SV48 defines four levels of page tables
        self.vpn0 = Signal(9)
        self.vpn1 = Signal(9)
        self.vpn2 = Signal(9)
        self.vpn3 = Signal(9)
        #TODO_PLATEN: use that signal
        self.is_2M = Signal()
        self.is_1G = Signal()
        self.valid = Signal()

    def flatten(self):
        return Cat(*self.ports())

    def eq(self, x):
        return self.flatten().eq(x.flatten())

    def ports(self):
        return [self.asid, self.vpn0, self.vpn1, self.vpn2,
                self.is_2M, self.is_1G, self.valid]

class TLBContent(Elaboratable):
    def __init__(self, pte_width, asid_width):
        self.asid_width = asid_width
        self.pte_width = pte_width
        self.flush_i = Signal()  # Flush signal
        # Update TLB
        self.update_i = TLBUpdate(asid_width)
        self.vpn2 = Signal(9)
        self.vpn1 = Signal(9)
        self.vpn0 = Signal(9)
        self.replace_en_i = Signal() # replace the following entry,
                                     # set by replacement strategy
        # Lookup signals
        self.lu_asid_i = Signal(asid_width)
        self.lu_content_o = Signal(pte_width)
        self.lu_is_2M_o = Signal()
        self.lu_is_1G_o = Signal()
        self.lu_hit_o = Signal()

    def elaborate(self, platform):
        m = Module()

        tags = TLBEntry(self.asid_width)
        content = Signal(self.pte_width)

        m.d.comb += [self.lu_hit_o.eq(0),
                     self.lu_is_2M_o.eq(0),
                     self.lu_is_1G_o.eq(0)]

        # temporaries for 1st level match
        asid_ok = Signal(reset_less=True)
        vpn2_ok = Signal(reset_less=True)
        tags_ok = Signal(reset_less=True)
        vpn2_hit = Signal(reset_less=True)
        m.d.comb += [tags_ok.eq(tags.valid),
                     asid_ok.eq(tags.asid == self.lu_asid_i),
                     vpn2_ok.eq(tags.vpn2 == self.vpn2),
                     vpn2_hit.eq(tags_ok & asid_ok & vpn2_ok)]
        # temporaries for 2nd level match
        vpn1_ok = Signal(reset_less=True)
        tags_2M = Signal(reset_less=True)
        vpn0_ok = Signal(reset_less=True)
        vpn0_or_2M = Signal(reset_less=True)
        m.d.comb += [vpn1_ok.eq(self.vpn1 == tags.vpn1),
                     tags_2M.eq(tags.is_2M),
                     vpn0_ok.eq(self.vpn0 == tags.vpn0),
                     vpn0_or_2M.eq(tags_2M | vpn0_ok)]
        # first level match, this may be a giga page,
        # check the ASID flags as well
        with m.If(vpn2_hit):
            # second level
            with m.If (tags.is_1G):
                m.d.comb += [ self.lu_content_o.eq(content),
                              self.lu_is_1G_o.eq(1),
                              self.lu_hit_o.eq(1),
                            ]
            # not a giga page hit so check further
            with m.Elif(vpn1_ok):
                # this could be a 2 mega page hit or a 4 kB hit
                # output accordingly
                with m.If(vpn0_or_2M):
                    m.d.comb += [ self.lu_content_o.eq(content),
                                  self.lu_is_2M_o.eq(tags.is_2M),
                                  self.lu_hit_o.eq(1),
                                ]
        # ------------------
        # Update or Flush
        # ------------------

        # temporaries
        replace_valid = Signal(reset_less=True)
        m.d.comb += replace_valid.eq(self.update_i.valid & self.replace_en_i)

        # flush
        with m.If (self.flush_i):
            # invalidate (flush) conditions: all if zero or just this ASID
            with m.If (self.lu_asid_i == Const(0, self.asid_width) |
                      (self.lu_asid_i == tags.asid)):
                m.d.sync += tags.valid.eq(0)

        # normal replacement
        with m.Elif(replace_valid):
            m.d.sync += [ # update tag array
                          tags.asid.eq(self.update_i.asid),
                          tags.vpn2.eq(self.update_i.vpn[18:27]),
                          tags.vpn1.eq(self.update_i.vpn[9:18]),
                          tags.vpn0.eq(self.update_i.vpn[0:9]),
                          tags.is_1G.eq(self.update_i.is_1G),
                          tags.is_2M.eq(self.update_i.is_2M),
                          tags.valid.eq(1),
                          # and content as well
                          content.eq(self.update_i.content.flatten())
                        ]
        return m

    def ports(self):
        return [self.flush_i,
                 self.lu_asid_i,
                 self.lu_is_2M_o, self.lu_is_1G_o, self.lu_hit_o,
                ] + self.update_i.content.ports() + self.update_i.ports()
