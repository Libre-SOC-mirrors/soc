from nmigen import Module, Signal
from nmigen.back.pysim import Simulator, Delay, Settle
from nmutil.formaltest import FHDLTestCase
import unittest
from soc.decoder.power_decoder import (create_pdecode)
from soc.decoder.power_enums import (Function, MicrOp,
                                     In1Sel, In2Sel, In3Sel,
                                     OutSel, RC, LdstLen, CryIn,
                                     single_bit_flags, Form,
                                     get_signal_name, get_csv)
from soc.decoder.power_decoder2 import (PowerDecode2)
from soc.simulator.program import Program
from soc.simulator.qemu import run_program
from soc.decoder.isa.all import ISA
from soc.fu.test.common import TestCase
from soc.config.endian import bigendian


class AttnTestCase(FHDLTestCase):
    test_data = []

    def __init__(self, name="general"):
        super().__init__(name)
        self.test_name = name

    def test_0_attn(self):
        """simple test of attn.  program is 4 long: should halt at 2nd op
        """
        lst = ["addi 6, 0, 0x10",
               "attn",
               "subf. 1, 6, 7",
               "cmp cr2, 1, 6, 7",
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1])

    def run_tst_program(self, prog, initial_regs=None, initial_sprs=None,
                        initial_mem=None):
        initial_regs = [0] * 32
        tc = TestCase(prog, self.test_name, initial_regs, initial_sprs, 0,
                      initial_mem, 0)
        self.test_data.append(tc)


class GeneralTestCases(FHDLTestCase):
    test_data = []

    def __init__(self, name="general"):
        super().__init__(name)
        self.test_name = name

    @unittest.skip("disable")
    def test_0_litex_bios_ctr_loop(self):
        """
        32a4:   ff ff 63 38     addi    r3,r3,-1
        32a8:   20 00 63 78     clrldi  r3,r3,32
        32ac:   01 00 23 39     addi    r9,r3,1
        32b0:   a6 03 29 7d     mtctr   r9
        32b4:   00 00 00 60     nop
        32b8:   fc ff 00 42     bdnz    32b4 <cdelay+0x10>
        32bc:   20 00 80 4e     blr

        notes on converting pseudo-assembler to actual:

        * bdnz target (equivalent to: bc 16,0,target)
        * Clear left immediate clrldi ra,rs,n (n < 64) rldicl ra,rs,0,n
        * CTR mtctr Rx mtspr 9,Rx
        """
        pass

    @unittest.skip("disable")
    def test_0_litex_bios_cmp(self):
        """litex bios cmp test
        """
        lst = [ "addis    26, 0, 21845",
                "ori      26, 26, 21845",
                "addi     5, 26, 0",
                "rldicr  5,5,32,31",
                "addi     5, 26, 0",
                "cmp     0, 0, 5, 26",
                "bc      12, 2, 28",
                "addis   6, 0, 1",
                "addis   7, 0, 1",
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [5,6,7,26], initial_mem={})

    @unittest.skip("disable")
    def test_0_litex_bios_r1(self):
        """litex bios IMM64 macro test
        """
        lst = [ "addis     1,0,0",
                 "ori     1,1,0",
                 "rldicr  1,1,32,31",
                 "oris    1,1,256",
                 "ori     1,1,3832",
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1], initial_mem={})

    @unittest.skip("disable")
    def test_0_litex_trampoline(self):
        lst = ["tdi   0,0,0x48",
               "b     0x28",
               "mfmsr r11",
               "bcl 20,31,4",
               "mflr r10",
               "addi r10,r10,20",
               "mthsrr0 r10",
               "mthsrr1 r11",
               "hrfid",
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [], initial_mem={})

    @unittest.skip("disable")
    def test_0_cmp(self):
        lst = ["addi 6, 0, 0x10",
               "addi 7, 0, 0x05",
               "subf. 1, 6, 7",
               "cmp cr2, 1, 6, 7",
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1])

    @unittest.skip("disable")
    def test_example(self):
        lst = ["addi 1, 0, 0x5678",
               "addi 2, 0, 0x1234",
               "add  3, 1, 2",
               "and  4, 1, 2"]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 2, 3, 4])

    @unittest.skip("disable")
    def test_ldst(self):
        lst = ["addi 1, 0, 0x5678",
               "addi 2, 0, 0x1234",
               "stw  1, 0(2)",
               "lwz  3, 0(2)"
               ]
        initial_mem = {0x1230: (0x5432123412345678, 8),
                       0x1238: (0xabcdef0187654321, 8),
                       }
        with Program(lst, bigendian) as program:
            self.run_tst_program(program,
                                 [1, 2, 3],
                                 initial_mem)

    @unittest.skip("disable")
    def test_ldst_update(self):
        lst = ["addi 1, 0, 0x5678",
               "addi 2, 0, 0x1234",
               "stwu  1, 0(2)",
               "lwz  3, 0(2)"
               ]
        initial_mem = {0x1230: (0x5432123412345678, 8),
                       0x1238: (0xabcdef0187654321, 8),
                       }
        with Program(lst, bigendian) as program:
            self.run_tst_program(program,
                                 [1, 2, 3],
                                 initial_mem)

    @unittest.skip("disable")
    def test_ld_rev_ext(self):
        lst = ["addi 1, 0, 0x5678",
               "addi 2, 0, 0x1234",
               "addi 4, 0, 0x40",
               "stw  1, 0x40(2)",
               "lwbrx  3, 4, 2"]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 2, 3])

    @unittest.skip("disable")
    def test_st_rev_ext(self):
        lst = ["addi 1, 0, 0x5678",
               "addi 2, 0, 0x1234",
               "addi 4, 0, 0x40",
               "stwbrx  1, 4, 2",
               "lwzx  3, 4, 2"]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 2, 3])

    @unittest.skip("disable")
    def test_ldst_extended(self):
        lst = ["addi 1, 0, 0x5678",
               "addi 2, 0, 0x1234",
               "addi 4, 0, 0x40",
               "stw  1, 0x40(2)",
               "lwzx  3, 4, 2"]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 2, 3])

    @unittest.skip("disable")
    def test_0_ldst_widths(self):
        lst = ["addis 1, 0, 0xdead",
               "ori 1, 1, 0xbeef",
               "addi 2, 0, 0x1000",
               "std 1, 0(2)",
               "lbz 1, 5(2)",
               "lhz 3, 4(2)",
               "lwz 4, 4(2)",
               "addi 5, 0, 0x12",
               "stb 5, 5(2)",
               "ld  5, 0(2)"]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 2, 3, 4, 5])

    @unittest.skip("disable")
    def test_sub(self):
        lst = ["addi 1, 0, 0x1234",
               "addi 2, 0, 0x5678",
               "subf 3, 1, 2",
               "subfic 4, 1, 0x1337",
               "neg 5, 1"]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 2, 3, 4, 5])

    @unittest.skip("disable")
    def test_add_with_carry(self):
        lst = ["addi 1, 0, 5",
               "neg 1, 1",
               "addi 2, 0, 7",
               "neg 2, 2",
               "addc 3, 2, 1",
               "addi 3, 3, 1"
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 2, 3])

    @unittest.skip("disable")
    def test_addis(self):
        lst = ["addi 1, 0, 0x0FFF",
               "addis 1, 1, 0x0F"
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1])

    @unittest.skip("broken")
    def test_mulli(self):
        lst = ["addi 1, 0, 3",
               "mulli 1, 1, 2"
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1])

    #@unittest.skip("disable")
    def test_crxor(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1008",
               "addi 3, 0, 0x01ee",
               "mtcrf 0b1111111, 3",
               "crxor 3, 30, 4",
               "mfcr 3",
               ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1008
        initial_regs[3] = 0x01ee
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [3, 4])

    #@unittest.skip("disable")
    def test_crxor_2(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1008",
               "addi 3, 0, 0x01ee",
               "mtcrf 0b1111111, 3",
               "crxor 29, 30, 29",
               "mfcr 3",
               ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1008
        initial_regs[3] = 0x01ee
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [3, 4])

    #@unittest.skip("disable")
    def test_crnand(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1008",
               "addi 3, 0, 0x01ee",
               "mtcrf 0b1111111, 3",
               "crnand 3, 30, 4",
               "mfcr 3",
               ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1008
        initial_regs[3] = 0x01ee
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [3, 4])

    #@unittest.skip("disable")
    def test_crnand_2(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1008",
               "addi 3, 0, 0x01ee",
               "mtcrf 0b1111111, 3",
               "crnand 28, 30, 29",
               "mfcr 3",
               ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1008
        initial_regs[3] = 0x01ee
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [3, 4])

    @unittest.skip("disable")
    def test_isel_1(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1008",
               "addi 3, 0, 0x01ee",
               "mtcrf 0b1111111, 3",
               "isel 4, 1, 2, 2"
               ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1008
        initial_regs[3] = 0x00ee
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [3, 4])

    #@unittest.skip("disable")
    def test_isel_2(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1008",
               "addi 3, 0, 0x01ee",
               "mtcrf 0b1111111, 3",
               "isel 4, 1, 2, 30"
               ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1008
        initial_regs[3] = 0x00ee
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [3, 4])

    @unittest.skip("disable")
    def test_isel_3(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1008",
               "addi 3, 0, 0x01ee",
               "mtcrf 0b1111111, 3",
               "isel 4, 1, 2, 31"
               ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1008
        initial_regs[3] = 0x00ee
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [3, 4])

    @unittest.skip("disable")
    def test_2_load_store(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1008",
               "addi 3, 0, 0x00ee",
               "stb 3, 1(2)",
               "lbz 4, 1(2)",
               ]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1008
        initial_regs[3] = 0x00ee
        initial_mem = {0x1000: (0x5432123412345678, 8),
                       0x1008: (0xabcdef0187654321, 8),
                       0x1020: (0x1828384822324252, 8),
                       }
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [3, 4], initial_mem)

    @unittest.skip("disable")
    def test_3_load_store(self):
        lst = ["addi 1, 0, 0x1004",
               "addi 2, 0, 0x1002",
               "addi 3, 0, 0x15eb",
               "sth 4, 0(2)",
               "lhz 4, 0(2)"]
        initial_regs = [0] * 32
        initial_regs[1] = 0x1004
        initial_regs[2] = 0x1002
        initial_regs[3] = 0x15eb
        initial_mem = {0x1000: (0x5432123412345678, 8),
                       0x1008: (0xabcdef0187654321, 8),
                       0x1020: (0x1828384822324252, 8),
                       }
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 2, 3, 4], initial_mem)

    @unittest.skip("disable")
    def test_nop(self):
        lst = ["addi 1, 0, 0x1004",
               "ori 0,0,0", # "preferred" form of nop
               "addi 3, 0, 0x15eb",
              ]
        initial_regs = [0] * 32
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [1, 3])

    @unittest.skip("disable")
    def test_zero_illegal(self):
        lst = bytes([0x10,0x00,0x20,0x39,
                     0x0,0x0,0x0,0x0,
                     0x0,0x0,0x0,0x0 ])
        disassembly = ["addi 9, 0, 0x10",
                       "nop", # not quite
                       "nop"] # not quite
        initial_regs = [0] * 32
        with Program(lst, bigendian) as program:
            program.assembly = '\n'.join(disassembly) + '\n' # XXX HACK!
            self.run_tst_program(program, [1, 3])

    def test_loop(self):
        """
        in godbolt.org:
        register unsigned long i asm ("r9");
        void square(void) {
            i = 16;
            do {
                i = i - 1;
            } while (i != 12);
        }
        """
        lst = ["addi 9, 0, 0x10",  # i = 16
               "addi 9,9,-1",    # i = i - 1
               "cmpi 2,1,9,12",     # compare 9 to value 12, store in CR2
               "bc 4,10,-8"         # branch if CR2 "test was != 12"
               ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [9], initial_mem={})

    @unittest.skip("disable")
    def test_30_addis(self):
        lst = [  # "addi 0, 0, 5",
            "addis 12, 0, 0",
        ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [12])

    @unittest.skip("disable")
    def test_31_addis(self):
        """tests for zero not in register zero
        """
        lst = [  "rldicr  0, 0, 32, 31",
                 "oris    0, 0, 32767",
                 "ori     0, 0, 65535",
                 "addis 1, 0, 1",
                 "ori     1, 1, 515",
                 "rldicr  1, 1, 32, 31",
                 "oris    1, 1, 1029",
                 "ori     1, 1, 1543",
                 "addis   2, 0, -1",
        ]
        with Program(lst, bigendian) as program:
            self.run_tst_program(program, [0, 1, 2])

    def run_tst_program(self, prog, initial_regs=None, initial_sprs=None,
                        initial_mem=None):
        initial_regs = [0] * 32
        tc = TestCase(prog, self.test_name, initial_regs, initial_sprs, 0,
                      initial_mem, 0)
        self.test_data.append(tc)


class DecoderBase:

    def run_tst(self, generator, initial_mem=None, initial_pc=0):
        m = Module()
        comb = m.d.comb

        gen = list(generator.generate_instructions())
        insn_code = generator.assembly.splitlines()
        instructions = list(zip(gen, insn_code))

        pdecode = create_pdecode()
        m.submodules.pdecode2 = pdecode2 = PowerDecode2(pdecode)

        # place program at requested address
        gen = (initial_pc, gen)

        simulator = ISA(pdecode2, [0] * 32, {}, 0, initial_mem, 0,
                        initial_insns=gen, respect_pc=True,
                        disassembly=insn_code,
                        initial_pc=initial_pc,
                        bigendian=bigendian)

        sim = Simulator(m)

        def process():
            # yield pdecode2.dec.bigendian.eq(bigendian)
            yield Settle()

            while True:
                try:
                    yield from simulator.setup_one()
                except KeyError:  # indicates instruction not in imem: stop
                    break
                yield Settle()
                yield from simulator.execute_one()
                yield Settle()

        sim.add_process(process)
        with sim.write_vcd("pdecode_simulator.vcd"):
            sim.run()

        return simulator

    def run_tst_program(self, prog, reglist, initial_mem=None,
                        extra_break_addr=None):
        import sys
        simulator = self.run_tst(prog, initial_mem=initial_mem,
                                 initial_pc=0x20000000)
        prog.reset()
        with run_program(prog, initial_mem, extra_break_addr,
                         bigendian=bigendian) as q:
            self.qemu_register_compare(simulator, q, reglist)
            self.qemu_mem_compare(simulator, q, True)
        print(simulator.gpr.dump())

    def qemu_mem_compare(self, sim, qemu, check=True):
        if False:  # disable convenient large interesting debugging memory dump
            addr = 0x0
            qmemdump = qemu.get_mem(addr, 2048)
            for i in range(len(qmemdump)):
                s = hex(int(qmemdump[i]))
                print("qemu mem %06x %s" % (addr+i*8, s))
        for k, v in sim.mem.mem.items():
            qmemdump = qemu.get_mem(k*8, 8)
            s = hex(int(qmemdump[0]))[2:]
            print("qemu mem %06x %16s" % (k*8, s))
        for k, v in sim.mem.mem.items():
            print("sim mem  %06x %016x" % (k*8, v))
        if not check:
            return
        for k, v in sim.mem.mem.items():
            qmemdump = qemu.get_mem(k*8, 1)
            self.assertEqual(int(qmemdump[0]), v)

    def qemu_register_compare(self, sim, qemu, regs):
        qpc, qxer, qcr = qemu.get_pc(), qemu.get_xer(), qemu.get_cr()
        sim_cr = sim.cr.value
        sim_pc = sim.pc.CIA.value
        sim_xer = sim.spr['XER'].value
        print("qemu pc", hex(qpc))
        print("qemu cr", hex(qcr))
        print("qemu xer", bin(qxer))
        print("sim nia", hex(sim.pc.NIA.value))
        print("sim pc", hex(sim.pc.CIA.value))
        print("sim cr", hex(sim_cr))
        print("sim xer", hex(sim_xer))
        self.assertEqual(qpc, sim_pc)
        for reg in regs:
            qemu_val = qemu.get_register(reg)
            sim_val = sim.gpr(reg).value
            self.assertEqual(qemu_val, sim_val,
                             "expect %x got %x" % (qemu_val, sim_val))
        self.assertEqual(qcr, sim_cr)


class DecoderTestCase(DecoderBase, GeneralTestCases):
    pass


if __name__ == "__main__":
    unittest.main()
