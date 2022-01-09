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

# linux kernel 5.7 first MMU enable
"""
                          rd @ 000bf803 di b000000000001033 sel ff 3.......
                          rd @ 000bf804 di                0 sel ff ........
                          rd @ 000bf805 di                0 sel ff ........
                          rd @ 000bf806 di            10000 sel ff ........
                          rd @ 000bf807 di c0000000005fc380 sel ff ........
                          rd @ 000bf800 di         80000000 sel ff ........
                          rd @ 000bf801 di c00000000059d400 sel ff ..Y.....
                          rd @ 000bf802 di c000000000000000 sel ff ........
pc     a588 insn 7c7a03a6 msr a000000000000003
pc     a58c insn 7c9b03a6 msr a000000000000003
pc     a590 insn 4c000024 msr a000000000000003
pc     a598 insn f82d0190 msr b000000000000033
                          rd @ 01c00000 di ad005c0000000040 sel ff ........
                          rd @ 01c00001 di                0 sel ff ........
                          rd @ 01c00002 di                0 sel ff ........
                          rd @ 01c00003 di                0 sel ff ........
                          rd @ 01c00004 di                0 sel ff ........
                          rd @ 01c00005 di                0 sel ff ........
                          rd @ 01c00006 di                0 sel ff ........
                          rd @ 01c00007 di                0 sel ff ........
                          rd @ 000b8000 di  9e0ff0f00000080 sel ff ........
                          rd @ 000b8001 di                0 sel ff ........
                          rd @ 000b8002 di                0 sel ff ........
                          rd @ 000b8003 di                0 sel ff ........
                          rd @ 000b8004 di                0 sel ff ........
                          rd @ 000b8005 di                0 sel ff ........
                          rd @ 000b8006 di                0 sel ff ........
                          rd @ 000b8007 di                0 sel ff ........
                          rd @ 01fffc00 di  9d0ff0f00000080 sel ff ........
                          rd @ 01fffc01 di                0 sel ff ........
                          rd @ 01fffc02 di                0 sel ff ........
                          rd @ 01fffc03 di                0 sel ff ........
                          rd @ 01fffc04 di                0 sel ff ........
                          rd @ 01fffc05 di                0 sel ff ........
                          rd @ 01fffc06 di                0 sel ff ........
                          rd @ 01fffc07 di                0 sel ff ........
                          rd @ 01fffa00 di 8f010000000000c0 sel ff ........
                          rd @ 01fffa01 di 8f012000000000c0 sel ff ........
                          rd @ 01fffa02 di 8f014000000000c0 sel ff ........
                          rd @ 01fffa03 di 8e016000000000c0 sel ff ........
                          rd @ 01fffa04 di 8e018000000000c0 sel ff ........
                          rd @ 01fffa05 di 8e01a000000000c0 sel ff ........
                          rd @ 01fffa06 di 8e01c000000000c0 sel ff ........
                          rd @ 01fffa07 di 8e01e000000000c0 sel ff ........
"""

microwatt_linux_5_7_boot = {
                  0x000bf803<<3: 0xb000000000001033,
                  0x000bf804<<3: 0x0,
                  0x000bf805<<3: 0x0,
                  0x000bf806<<3: 0x10000,
                  0x000bf807<<3: 0xc0000000005fc380,
                  0x000bf800<<3: 0x80000000,
                  0x000bf801<<3: 0xc00000000059d400,
                  0x000bf802<<3: 0xc000000000000000,
                  0x01c00000<<3: 0xad005c0000000040,
                  0x01c00001<<3: 0x0,
                  0x01c00002<<3: 0x0,
                  0x01c00003<<3: 0x0,
                  0x01c00004<<3: 0x0,
                  0x01c00005<<3: 0x0,
                  0x01c00006<<3: 0x0,
                  0x01c00007<<3: 0x0,
                  0x000b8000<<3: 0x09e0ff0f00000080,
                  0x000b8001<<3: 0x0,
                  0x000b8002<<3: 0x0,
                  0x000b8003<<3: 0x0,
                  0x000b8004<<3: 0x0,
                  0x000b8005<<3: 0x0,
                  0x000b8006<<3: 0x0,
                  0x000b8007<<3: 0x0,
                  0x01fffc00<<3: 0x09d0ff0f00000080,
                  0x01fffc01<<3: 0x0,
                  0x01fffc02<<3: 0x0,
                  0x01fffc03<<3: 0x0,
                  0x01fffc04<<3: 0x0,
                  0x01fffc05<<3: 0x0,
                  0x01fffc06<<3: 0x0,
                  0x01fffc07<<3: 0x0,
                  0x01fffa00<<3: 0x8f010000000000c0,
                  0x01fffa01<<3: 0x8f012000000000c0,
                  0x01fffa02<<3: 0x8f014000000000c0,
                  0x01fffa03<<3: 0x8e016000000000c0,
                  0x01fffa04<<3: 0x8e018000000000c0,
                  0x01fffa05<<3: 0x8e01a000000000c0,
                  0x01fffa06<<3: 0x8e01c000000000c0,
                  0x01fffa07<<3: 0x8e01e000000000c0,
}
