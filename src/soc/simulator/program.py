# License: LGPLv3+
# Copyright (C) 2020 Michael Nolan <mtnolan2640@gmail.com>
# Copyright (C) 2020 Luke Kenneth Casson Leighton <lkcl@lkcl.net>

"""POWER Program

takes powerpc assembly instructions and turns them into LE/BE binary
data.  calls powerpc64-linux-gnu-as, ld and objcopy to do so.
"""

import tempfile
import subprocess
import struct
import os
import sys
from io import BytesIO

from soc.simulator.envcmds import cmds

filedir = os.path.dirname(os.path.realpath(__file__))
memmap = os.path.join(filedir, "memmap")


class Program:
    def __init__(self, instructions, bigendian):
        self.bigendian = bigendian
        if self.bigendian:
            self.endian_fmt = "elf64-big"
            self.obj_fmt = "-be"
            self.ld_fmt = "-EB"
        else:
            self.ld_fmt = "-EL"
            self.endian_fmt = "elf64-little"
            self.obj_fmt = "-le"

        if isinstance(instructions, bytes):  # actual bytes
            self.binfile = BytesIO(instructions)
            self.binfile.name = "assembly"
            self.assembly = ''  # noo disassemble number fiiive
        elif isinstance(instructions, str):  # filename
            # read instructions into a BytesIO to avoid "too many open files"
            with open(instructions, "rb") as f:
                b = f.read()
            self.binfile = BytesIO(b)
            self.assembly = ''  # noo disassemble number fiiive
            print("program", self.binfile)
        else:
            if isinstance(instructions, list):
                instructions = '\n'.join(instructions)
            self.assembly = instructions + '\n'  # plus final newline
            self._assemble()
        self._instructions = list(self._get_instructions())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _get_binary(self, elffile):
        self.binfile = tempfile.NamedTemporaryFile(suffix=".bin")
        args = [cmds['objcopy'],
                "-O", "binary",
                "-I", self.endian_fmt,
                elffile.name,
                self.binfile.name]
        subprocess.check_output(args)

    def _link(self, ofile):
        with tempfile.NamedTemporaryFile(suffix=".elf") as elffile:
            args = [cmds['ld'],
                    self.ld_fmt,
                    "-o", elffile.name,
                    "-T", memmap,
                    ofile.name]
            subprocess.check_output(args)
            self._get_binary(elffile)

    def _assemble(self):
        with tempfile.NamedTemporaryFile(suffix=".o") as outfile:
            args = [cmds['as'],
                    '-mpower9',
                    '-mregnames',
                    self.obj_fmt,
                    "-o",
                    outfile.name]
            p = subprocess.Popen(args, stdin=subprocess.PIPE)
            p.communicate(self.assembly.encode('utf-8'))
            if p.wait() != 0:
                print("Error in program:")
                print(self.assembly)
                sys.exit(1)
            self._link(outfile)

    def _get_instructions(self):
        while True:
            data = self.binfile.read(4)
            if not data:
                break
            yield struct.unpack('<I', data)[0]  # unsigned int

    def generate_instructions(self):
        yield from self._instructions

    def reset(self):
        self.binfile.seek(0)

    def size(self):
        curpos = self.binfile.tell()
        self.binfile.seek(0, 2)  # Seek to end of file
        size = self.binfile.tell()
        self.binfile.seek(curpos, 0)
        return size

    def write_bin(self, fname):
        self.reset()
        data = self.binfile.read()
        with open(fname, "wb") as f:
            f.write(data)

    def close(self):
        self.binfile.close()

if __name__ == '__main__':
    lst = ['addi 5, 0, 4660/2',
           'mtcrf 255, 5+3',
           'mfocrf 2, 1',
           'addi r2, 3, 1',
           'attn',
          ]
    lst = ["addi 9, 0, 0x10",  # i = 16
           "addi 9,9,-1",    # i = i - 1
           "cmpi 2,1,9,12",     # compare 9 to value 12, store in CR2
           "bc 4,10,-8",        # branch if CR2 "test was != 12"
           'attn',
           ]

    with Program(lst, False) as p:
        for instruction in p.generate_instructions():
            print (hex(instruction))
        p.write_bin("/tmp/test.bin")
