def b(x): # byte-reverse function
    return int.from_bytes(x.to_bytes(8, byteorder='little'),
                          byteorder='big', signed=False)

test1 = {
           0x10000:    # PARTITION_TABLE_2
                       # PATB_GR=1 PRTB=0x1000 PRTS=0xb
           b(0x800000000100000b),

           0x30000:     # RADIX_ROOT_PTE
                        # V = 1 L = 0 NLB = 0x400 NLS = 9
           b(0x8000000000040009),

           0x40000:     # RADIX_SECOND_LEVEL
                        # V = 1 L = 1 SW = 0 RPN = 0
                        # R = 1 C = 1 ATT = 0 EAA 0x3
           b(0xc000000000000183),

           0x1000000:   # PROCESS_TABLE_3
                        # RTS1 = 0x2 RPDB = 0x300 RTS2 = 0x5 RPDS = 13
           b(0x40000000000300ad),

           #0x10004: 0

}


# executable permission is barred here (EAA=0x2)
test2 = {
           0x10000:    # PARTITION_TABLE_2
                       # PATB_GR=1 PRTB=0x1000 PRTS=0xb
           b(0x800000000100000b),

           0x30000:     # RADIX_ROOT_PTE
                        # V = 1 L = 0 NLB = 0x400 NLS = 9
           b(0x8000000000040009),

           0x40000:     # RADIX_SECOND_LEVEL
                        # V = 1 L = 1 SW = 0 RPN = 0
                        # R = 1 C = 1 ATT = 0 EAA 0x2
           b(0xc000000000000182),

           0x1000000:   # PROCESS_TABLE_3
                        # RTS1 = 0x2 RPDB = 0x300 RTS2 = 0x5 RPDS = 13
           b(0x40000000000300ad),

           #0x10004: 0

}


# microwatt mmu.bin first part of test 2. PRTBL must be set to 0x12000, PID to 1
microwatt_test2 = {
             0x13920: 0x86810000000000c0, # leaf node
             0x10000: 0x0930010000000080, # directory node
             0x12010: 0x0a00010000000000, # page table
             0x8108: 0x0000000badc0ffee,  # memory to be looked up
            }

microwatt_test4 = {
             0x13858: 0x86a10000000000c0, # leaf node
             0x10000: 0x0930010000000080, # directory node
             0x12010: 0x0a00010000000000, # page table
}

# microwatt mmu.bin test 5: a misaligned read which crosses over to a TLB that
# is not valid.  must attempt a 64-bit read at address 0x39fffd to trigger

microwatt_test5 = {
             0x13cf8: 0x86b10000000000c0, # leaf, covers up to 0x39ffff
             0x10008: 0x0930010000000080, # directory node
             0x12010: 0x0a00010000000000, # page table
             0x39fff8: 0x0123456badc0ffee,  # to be looked up (should fail)
             0x400000: 0x0123456badc0ffee,  # not page-mapped
}

