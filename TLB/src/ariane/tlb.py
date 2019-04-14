"""
# Copyright 2018 ETH Zurich and University of Bologna.
# Copyright and related rights are licensed under the Solderpad Hardware
# License, Version 0.51 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
# http:#solderpad.org/licenses/SHL-0.51. Unless required by applicable law
# or agreed to in writing, software, hardware and materials distributed under
# this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# Author: David Schaffenrath, TU Graz
# Author: Florian Zaruba, ETH Zurich
# Date: 21.4.2017
# Description: Translation Lookaside Buffer, SV39
#              fully set-associative
"""
from math import log

# SV39 defines three levels of page tables
class TLBEntry:
    def __init__(self):
        self.asid = Signal(ASID_WIDTH)
        self.vpn2 = Signal(9)
        self.vpn1 = Signal(9)
        self.vpn0 = Signal(9)
        self.is_2M = Signal()
        self.is_1G = Signal()
        self.valid = Signal()

TLB_ENTRIES = 4
ASID_WIDTH  = 1

from .tlb import TLBUpdate, PTE

module tlb #(
  )(
    flush_i = Signal(),  # Flush signal
    # Update TLB
    update_i = TLBUpdate()
    # Lookup signals
    lu_access_i = Signal()
    lu_asid_i = Signal(ASID_WIDTH)
    lu_vaddr_i = Signal(64)
    lu_content_o = PTE()
    lu_is_2M_o = Signal()
    lu_is_1G_o = Signal()
    lu_hit_o = Signal()
)

    tags = TLBEntry()
    # SV39 defines three levels of page tables

    content = Array([TLB() for i in range(TLB_ENTRIES)])

    vpn2 = Signal(9)
    vpn1 = Signal(9)
    vpn0 = Signal(9)
    lu_hit = Signal(TLB_ENTRIES)     # to replacement logic
    replace_en = Signal(TLB_ENTRIES) # replace the following entry,
                                     # set by replacement strategy
    #-------------
    # Translation
    #-------------
        m.d.comb += [ vpn0.eq(lu_vaddr_i[12:21],
                      vpn1.eq(lu_vaddr_i[21:30],
                      vpn2.eq(lu_vaddr_i[30:39]
                    ]

        for i in range(TLB_ENTRIES):
            m.d.comb += lu_hit[i].eq(0)
            # first level match, this may be a giga page,
            # check the ASID flags as well
            with m.If(tags[i].valid & \
                      tags[i].asid == lu_asid_i &  \
                      tags[i].vpn2 == vpn2):
                # second level
                with m.If (tags[i].is_1G):
                    m.d.sync += lu_content_o.eq(content[i])
                    m.d.comb += [ lu_is_1G_o.eq(1),
                                  lu_hit_o.eq(1),
                                  lu_hit[i].eq(1),
                                ]
                # not a giga page hit so check further
                with m.Elif(vpn1 == tags[i].vpn1):
                    # this could be a 2 mega page hit or a 4 kB hit
                    # output accordingly
                    with m.If(tags[i].is_2M | vpn0 == tags[i].vpn0):
                        m.d.sync += lu_content_o.eq(content[i])
                        m.d.comb += [ lu_is_2M_o.eq(tags[i].is_2M),
                                      lu_hit_o.eq(1),
                                      lu_hit[i].eq(1),
                                    ]

    # ------------------
    # Update and Flush
    # ------------------

        for i in range(TLB_ENTRIES):
            with m.If (flush_i):
                # invalidate logic: flush conditions
                with m.If ((lu_asid_i == Const(0, ASID_WIDTH) | # all if zero
                           (lu_asid_i == tags[i].asid):       # just this ASID
                    m.d.sync += tags[i].valid.eq(0)

            # normal replacement
            with m.Elif(update_i.valid & replace_en[i]):
                m.d.sync += [ # update tag array
                              tags[i].asid.eq(update_i.asid),
                              tags[i].vpn2.eq(update_i.vpn [18:27]),
                              tags[i].vpn1.eq(update_i.vpn [9:18]),
                              tags[i].vpn0.eq(update_i.vpn[0:9]),
                              tags[i].is_1G.eq(update_i.is_1G),
                              tags[i].is_2M.eq(update_i.is_2M),
                              tags[i].valid.eq(1),
                              # and content as well
                              content[i].eq(update_i.content)
                            ]

    # -----------------------------------------------
    # PLRU - Pseudo Least Recently Used Replacement
    # -----------------------------------------------
        TLBSZ = 2*(TLB_ENTRIES-1)
        plru_tree = Signal(TLBSZ)

        # The PLRU-tree indexing:
        # lvl0        0
        #            / \
        #           /   \
        # lvl1     1     2
        #         / \   / \
        # lvl2   3   4 5   6
        #       / \ /\/\  /\
        #      ... ... ... ...
        # Just predefine which nodes will be set/cleared
        # E.g. for a TLB with 8 entries, the for-loop is semantically
        # equivalent to the following pseudo-code:
        # unique case (1'b1)
        # lu_hit[7]: plru_tree[0, 2, 6] = {1, 1, 1};
        # lu_hit[6]: plru_tree[0, 2, 6] = {1, 1, 0};
        # lu_hit[5]: plru_tree[0, 2, 5] = {1, 0, 1};
        # lu_hit[4]: plru_tree[0, 2, 5] = {1, 0, 0};
        # lu_hit[3]: plru_tree[0, 1, 4] = {0, 1, 1};
        # lu_hit[2]: plru_tree[0, 1, 4] = {0, 1, 0};
        # lu_hit[1]: plru_tree[0, 1, 3] = {0, 0, 1};
        # lu_hit[0]: plru_tree[0, 1, 3] = {0, 0, 0};
        # default: begin /* No hit */ end
        # endcase
        LOG_TLB = int(log2(TLB_ENTRIES))
        for i in range(TLB_ENTRIES):
            # we got a hit so update the pointer as it was least recently used
            with m.If (lu_hit[i] & lu_access_i):
                # Set the nodes to the values we would expect
                for lvl in range(LOG_TLB):
                    idx_base = (1<<lvl)-1
                    # lvl0 <=> MSB, lvl1 <=> MSB-1, ...
                    shift = LOG_TLB - lvl;
                    new_idx = Const(~((i >> (shift-1)) & 1)
                    m.d.sync += plru_tree[idx_base + (i >> shift)].eq(new_idx)

        # Decode tree to write enable signals
        # Next for-loop basically creates the following logic for e.g.
        # an 8 entry TLB (note: pseudo-code obviously):
        # replace_en[7] = &plru_tree[ 6, 2, 0]; #plru_tree[0,2,6]=={1,1,1}
        # replace_en[6] = &plru_tree[~6, 2, 0]; #plru_tree[0,2,6]=={1,1,0}
        # replace_en[5] = &plru_tree[ 5,~2, 0]; #plru_tree[0,2,5]=={1,0,1}
        # replace_en[4] = &plru_tree[~5,~2, 0]; #plru_tree[0,2,5]=={1,0,0}
        # replace_en[3] = &plru_tree[ 4, 1,~0]; #plru_tree[0,1,4]=={0,1,1}
        # replace_en[2] = &plru_tree[~4, 1,~0]; #plru_tree[0,1,4]=={0,1,0}
        # replace_en[1] = &plru_tree[ 3,~1,~0]; #plru_tree[0,1,3]=={0,0,1}
        # replace_en[0] = &plru_tree[~3,~1,~0]; #plru_tree[0,1,3]=={0,0,0}
        # For each entry traverse the tree. If every tree-node matches,
        # the corresponding bit of the entry's index, this is
        # the next entry to replace.
        for i in range(TLB_ENTRIES):
            en = [Const(1)]
            for lvl in range(LOG_TLB):
                idx_base = (1<<lvl)-1
                # lvl0 <=> MSB, lvl1 <=> MSB-1, ...
                shift = LOG_TLB - lvl;
                new_idx = (i >> (shift-1)) & 1;
                plru = plru_tree[idx_base + (i>>shift)]
                if new_idx:
                    en.append(~plru) # yes inverted (using bool())
                else:
                    en.append(plru)  # yes inverted (using bool())
            # this is equivalent to plru0 & plru1 & plru2 ...
            # bool() is an *OR*, so invert individual items, OR, then invert,
            # and it becomes an AND of the concatenated list of bits
            m.d.sync += replace_en[i].eq(~Cat(*en).bool())

    #--------------
    # Sanity checks
    #--------------

    assert (TLB_ENTRIES % 2 == 0) and (TLB_ENTRIES > 1)), \
        "TLB size must be a multiple of 2 and greater than 1"
    assert (ASID_WIDTH >= 1),
        "ASID width must be at least 1")

    """
    # Just for checking
    function int countSetBits(logic[TLB_ENTRIES-1:0] vector);
      automatic int count = 0;
      foreach (vector[idx]) begin
        count += vector[idx];
      end
      return count;
    endfunction

    assert property (@(posedge clk_i)(countSetBits(lu_hit) <= 1))
      else begin $error("More then one hit in TLB!"); $stop(); end
    assert property (@(posedge clk_i)(countSetBits(replace_en) <= 1))
      else begin $error("More then one TLB entry selected for next replace!");
    """

