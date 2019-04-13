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
# Date: 24.4.2017
# Description: Hardware-PTW

/* verilator lint_off WIDTH */
import ariane_pkg::*;
"""

class DCacheReqI:
    def __init__(self):
        self.address_index = Signal(DCACHE_INDEX_WIDTH)
        self.address_tag = Signal(DCACHE_TAG_WIDTH)
        self.data_wdata = Signal(64)
        self.data_req = Signal()
        self.data_we = Signal()
        self.data_be = Signal(8)
        self.data_size = Signal(2)
        self.kill_req = Signal()
        self.tag_valid = Signal()


class DCacheReqO:
    def __init__(self):
        data_gnt = Signal()
        data_rvalid = Signal()
        data_rdata = Signal(64)

ASID_WIDTH = 1

class PTE(RecordObject):
    def __init__(self):
        self.reserved = Signal(10)
        self.ppn = Signal(44)
        self.rsw = Signal(2)
        self.d = Signal()
        self.a = Signal()
        self.g = Signal()
        self.u = Signal()
        self.x = Signal()
        self.w = Signal()
        self.r = Signal()
        self.v = Signal()


class TLBUpdate:
    def __init__(self):
        valid = Signal()      # valid flag
        is_2M = Signal() 
        is_1G = Signal() 
        vpn = Signal(27)
        asid = Signal(ASID_WIDTH)
        content = PTE()

IDLE = 0
WAIT_GRANT = 1
PTE_LOOKUP = 2
WAIT_RVALID = 3
PROPAGATE_ERROR = 4

# SV39 defines three levels of page tables
LVL1 = Const(0, 2)
LVL2 = Const(1, 2)
LVL3 = Const(2, 2)


class PTW:
    flush_i = Signal() # flush everything, we need to do this because
    # actually everything we do is speculative at this stage
    # e.g.: there could be a CSR instruction that changes everything
    ptw_active_o = Signal()
    walking_instr_o = Signal()        # set when walking for TLB
    ptw_error_o = Signal()            # set when an error occurred
    enable_translation_i = Signal()   # CSRs indicate to enable SV39
    en_ld_st_translation_i = Signal() # enable VM translation for load/stores

    lsu_is_store_i = Signal() ,       # this translation triggered by a store
    # PTW memory interface
    req_port_i = DCacheReqO()
    req_port_o = DCacheReqI()

    # to TLBs, update logic
    itlb_update_o = TLBUpdate()
    dtlb_update_o = TLBUpdate()

    update_vaddr_o = Signal(39)

    asid_i = Signal(ASID_WIDTH)
    # from TLBs
    # did we miss?
    itlb_access_i = Signal()
    itlb_hit_i = Signal()
    itlb_vaddr_i = Signal(64)

    dtlb_access_i = Signal()
    dtlb_hit_i = Signal()
    dtlb_vaddr_i = Signal(64)
    # from CSR file
    satp_ppn_i = Signal(44) # ppn from satp
    mxr_i = Signal()
    # Performance counters
    itlb_miss_o = Signal()
    dtlb_miss_o = Signal()

);

    # input registers
    data_rvalid_q = Signal()
    data_rdata_q = Signal(64)

    pte = PTE()
    assign pte = riscv::pte_t(data_rdata_q);

    # is this an instruction page table walk?
    is_instr_ptw_q = Signal()
    is_instr_ptw_n = Signal()
    global_mapping_q = Signal()
    global_mapping_n = Signal()
    # latched tag signal
    tag_valid_n = Signal()
    tag_valid_q = Signal()
    # register the ASID
    tlb_update_asid_q = Signal(ASID_WIDTH)
    tlb_update_asid_n = Signal(ASID_WIDTH)
    # register the VPN we need to walk, SV39 defines a 39 bit virtual address
    vaddr_q = Signal(64)
    vaddr_n = Signal(64)
    # 4 byte aligned physical pointer
    ptw_pptr_q = Signal(56)
    ptw_pptr_n = Signal(56)

    # Assignments
    m.d.comb += update_vaddr_o.eq(vaddr_q)

    m.d.comb += ptw_active_o.eq(state_q != IDLE)
    m.d.comb += walking_instr_o.eq(is_instr_ptw_q)
    # directly output the correct physical address
    m.d.comb += req_port_o.address_index.eq(ptw_pptr_q[0:DCACHE_INDEX_WIDTH])
    end = DCACHE_INDEX_WIDTH + DCACHE_TAG_WIDTH
    m.d.comb += req_port_o.address_tag.eq(ptw_pptr_q[DCACHE_INDEX_WIDTH:end])
    # we are never going to kill this request
    m.d.comb += req_port_o.kill_req.eq(0)
    # we are never going to write with the HPTW
    m.d.comb += req_port_o.data_wdata.eq(Const(0, 64))
    # -----------
    # TLB Update
    # -----------
    assign itlb_update_o.vpn = vaddr_q[38:12];
    assign dtlb_update_o.vpn = vaddr_q[38:12];
    # update the correct page table level
    assign itlb_update_o.is_2M = (ptw_lvl_q == LVL2);
    assign itlb_update_o.is_1G = (ptw_lvl_q == LVL1);
    assign dtlb_update_o.is_2M = (ptw_lvl_q == LVL2);
    assign dtlb_update_o.is_1G = (ptw_lvl_q == LVL1);
    # output the correct ASID
    assign itlb_update_o.asid = tlb_update_asid_q;
    assign dtlb_update_o.asid = tlb_update_asid_q;
    # set the global mapping bit
    assign itlb_update_o.content = pte | (global_mapping_q << 5);
    assign dtlb_update_o.content = pte | (global_mapping_q << 5);

    assign req_port_o.tag_valid      = tag_valid_q;

    #-------------------
    # Page table walker
    #-------------------
    # A virtual address va is translated into a physical address pa as follows:
    # 1. Let a be sptbr.ppn × PAGESIZE, and let i = LEVELS-1. (For Sv39,
    #    PAGESIZE=2^12 and LEVELS=3.)
    # 2. Let pte be the value of the PTE at address a+va.vpn[i]×PTESIZE. (For
    #    Sv32, PTESIZE=4.)
    # 3. If pte.v = 0, or if pte.r = 0 and pte.w = 1, stop and raise an access
    #    exception.
    # 4. Otherwise, the PTE is valid. If pte.r = 1 or pte.x = 1, go to step 5.
    #    Otherwise, this PTE is a pointer to the next level of the page table.
    #    Let i=i-1. If i < 0, stop and raise an access exception. Otherwise, let
    #    a = pte.ppn × PAGESIZE and go to step 2.
    # 5. A leaf PTE has been found. Determine if the requested memory access
    #    is allowed by the pte.r, pte.w, and pte.x bits. If not, stop and
    #    raise an access exception. Otherwise, the translation is successful.
    #    Set pte.a to 1, and, if the memory access is a store, set pte.d to 1.
    #    The translated physical address is given as follows:
    #      - pa.pgoff = va.pgoff.
    #      - If i > 0, then this is a superpage translation and
    #        pa.ppn[i-1:0] = va.vpn[i-1:0].
    #      - pa.ppn[LEVELS-1:i] = pte.ppn[LEVELS-1:i].
    always_comb begin : ptw
        # default assignments
        # PTW memory interface
        tag_valid_n           = 1'b0;
        req_port_o.data_req   = 1'b0;
        req_port_o.data_be    = 8'hFF;
        req_port_o.data_size  = 2'b11;
        req_port_o.data_we    = 1'b0;
        ptw_error_o           = 1'b0;
        itlb_update_o.valid   = 1'b0;
        dtlb_update_o.valid   = 1'b0;
        is_instr_ptw_n        = is_instr_ptw_q;
        ptw_lvl_n             = ptw_lvl_q;
        ptw_pptr_n            = ptw_pptr_q;
        state_d               = state_q;
        global_mapping_n      = global_mapping_q;
        # input registers
        tlb_update_asid_n     = tlb_update_asid_q;
        vaddr_n               = vaddr_q;

        itlb_miss_o           = 1'b0;
        dtlb_miss_o           = 1'b0;

        case (state_q)

            IDLE: begin
                # by default we start with the top-most page table
                ptw_lvl_n        = LVL1;
                global_mapping_n = 1'b0;
                is_instr_ptw_n   = 1'b0;
                # if we got an ITLB miss
                if (enable_translation_i & itlb_access_i & ~itlb_hit_i & ~dtlb_access_i) begin
                    ptw_pptr_n          = {satp_ppn_i, itlb_vaddr_i[38:30], 3'b0};
                    is_instr_ptw_n      = 1'b1;
                    tlb_update_asid_n   = asid_i;
                    vaddr_n             = itlb_vaddr_i;
                    state_d             = WAIT_GRANT;
                    itlb_miss_o         = 1'b1;
                # we got an DTLB miss
                end else if (en_ld_st_translation_i & dtlb_access_i & ~dtlb_hit_i) begin
                    ptw_pptr_n          = {satp_ppn_i, dtlb_vaddr_i[38:30], 3'b0};
                    tlb_update_asid_n   = asid_i;
                    vaddr_n             = dtlb_vaddr_i;
                    state_d             = WAIT_GRANT;
                    dtlb_miss_o         = 1'b1;
                end
            end

            WAIT_GRANT: begin
                # send a request out
                req_port_o.data_req = 1'b1;
                # wait for the WAIT_GRANT
                if (req_port_i.data_gnt) begin
                    # send the tag valid signal one cycle later
                    tag_valid_n = 1'b1;
                    state_d     = PTE_LOOKUP;
                end
            end

            PTE_LOOKUP: begin
                # we wait for the valid signal
                if (data_rvalid_q) begin

                    # check if the global mapping bit is set
                    if (pte.g)
                        global_mapping_n = 1'b1;

                    # -------------
                    # Invalid PTE
                    # -------------
                    # If pte.v = 0, or if pte.r = 0 and pte.w = 1, stop and raise a page-fault exception.
                    if (!pte.v || (!pte.r && pte.w))
                        state_d = PROPAGATE_ERROR;
                    # -----------
                    # Valid PTE
                    # -----------
                    else begin
                        state_d = IDLE;
                        # it is a valid PTE
                        # if pte.r = 1 or pte.x = 1 it is a valid PTE
                        if (pte.r || pte.x) begin
                            # Valid translation found (either 1G, 2M or 4K entry)
                            if (is_instr_ptw_q) begin
                                # ------------
                                # Update ITLB
                                # ------------
                                # If page is not executable, we can directly raise an error. This
                                # doesn't put a useless entry into the TLB. The same idea applies
                                # to the access flag since we let the access flag be managed by SW.
                                if (!pte.x || !pte.a)
                                  state_d = PROPAGATE_ERROR;
                                else
                                  itlb_update_o.valid = 1'b1;

                            end else begin
                                # ------------
                                # Update DTLB
                                # ------------
                                # Check if the access flag has been set, otherwise throw a page-fault
                                # and let the software handle those bits.
                                # If page is not readable (there are no write-only pages)
                                # we can directly raise an error. This doesn't put a useless
                                # entry into the TLB.
                                if (pte.a && (pte.r || (pte.x && mxr_i))) begin
                                  dtlb_update_o.valid = 1'b1;
                                end else begin
                                  state_d   = PROPAGATE_ERROR;
                                end
                                # Request is a store: perform some additional checks
                                # If the request was a store and the page is not write-able, raise an error
                                # the same applies if the dirty flag is not set
                                if (lsu_is_store_i && (!pte.w || !pte.d)) begin
                                    dtlb_update_o.valid = 1'b0;
                                    state_d   = PROPAGATE_ERROR;
                                end
                            end
                            # check if the ppn is correctly aligned:
                            # 6. If i > 0 and pa.ppn[i − 1 : 0] != 0, this is a misaligned superpage; stop and raise a page-fault
                            # exception.
                            if (ptw_lvl_q == LVL1 && pte.ppn[17:0] != '0) begin
                                state_d             = PROPAGATE_ERROR;
                                dtlb_update_o.valid = 1'b0;
                                itlb_update_o.valid = 1'b0;
                            end else if (ptw_lvl_q == LVL2 && pte.ppn[8:0] != '0) begin
                                state_d             = PROPAGATE_ERROR;
                                dtlb_update_o.valid = 1'b0;
                                itlb_update_o.valid = 1'b0;
                            end
                        # this is a pointer to the next TLB level
                        end else begin
                            # pointer to next level of page table
                            if (ptw_lvl_q == LVL1) begin
                                # we are in the second level now
                                ptw_lvl_n  = LVL2;
                                ptw_pptr_n = {pte.ppn, vaddr_q[29:21], 3'b0};
                            end

                            if (ptw_lvl_q == LVL2) begin
                                # here we received a pointer to the third level
                                ptw_lvl_n  = LVL3;
                                ptw_pptr_n = {pte.ppn, vaddr_q[20:12], 3'b0};
                            end

                            state_d = WAIT_GRANT;

                            if (ptw_lvl_q == LVL3) begin
                              # Should already be the last level page table => Error
                              ptw_lvl_n   = LVL3;
                              state_d = PROPAGATE_ERROR;
                            end
                        end
                    end
                end
                # we've got a data WAIT_GRANT so tell the cache that the tag is valid
            end
            # Propagate error to MMU/LSU
            PROPAGATE_ERROR: begin
                state_d     = IDLE;
                ptw_error_o = 1'b1;
            end
            # wait for the rvalid before going back to IDLE
            WAIT_RVALID: begin
                if (data_rvalid_q)
                    state_d = IDLE;
            end
            default: begin
                state_d = IDLE;
            end
        endcase

        # -------
        # Flush
        # -------
        # should we have flushed before we got an rvalid, wait for it until going back to IDLE
        if (flush_i) begin
            # on a flush check whether we are
            # 1. in the PTE Lookup check whether we still need to wait for an rvalid
            # 2. waiting for a grant, if so: wait for it
            # if not, go back to idle
            if ((state_q == PTE_LOOKUP && !data_rvalid_q) || ((state_q == WAIT_GRANT) && req_port_i.data_gnt))
                state_d = WAIT_RVALID;
            else
                state_d = IDLE;
        end
    end

    # sequential process
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (~rst_ni) begin
            state_q            <= IDLE;
            is_instr_ptw_q     <= 1'b0;
            ptw_lvl_q          <= LVL1;
            tag_valid_q        <= 1'b0;
            tlb_update_asid_q  <= '0;
            vaddr_q            <= '0;
            ptw_pptr_q         <= '0;
            global_mapping_q   <= 1'b0;
            data_rdata_q       <= '0;
            data_rvalid_q      <= 1'b0;
        end else begin
            state_q            <= state_d;
            ptw_pptr_q         <= ptw_pptr_n;
            is_instr_ptw_q     <= is_instr_ptw_n;
            ptw_lvl_q          <= ptw_lvl_n;
            tag_valid_q        <= tag_valid_n;
            tlb_update_asid_q  <= tlb_update_asid_n;
            vaddr_q            <= vaddr_n;
            global_mapping_q   <= global_mapping_n;
            data_rdata_q       <= req_port_i.data_rdata;
            data_rvalid_q      <= req_port_i.data_rvalid;
        end
    end

endmodule
/* verilator lint_on WIDTH */
