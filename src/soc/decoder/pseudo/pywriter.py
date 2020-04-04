# python code-writer for OpenPOWER ISA pseudo-code parsing

import os
from soc.decoder.pseudo.pagereader import ISA
from soc.decoder.power_pseudo import convert_to_python
from soc.decoder.orderedset import OrderedSet

def get_isasrc_dir():
    fdir = os.path.abspath(os.path.dirname(__file__))
    fdir = os.path.split(fdir)[0]
    return os.path.join(fdir, "isa")

def create_args(reglist, extra=None):
    args = OrderedSet()
    for reg in reglist:
        args.add(reg)
    args = list(args)
    if extra:
        args = [extra] + args
    return ', '.join(args)


header = """\
# auto-generated by pywriter.py, do not edit or commit

from soc.decoder.isa.caller import ISACaller, inject
from soc.decoder.helpers import (EXTS64, EXTZ64, ROTL64, ROTL32, MASK,)
from soc.decoder.selectable_int import SelectableInt
from soc.decoder.selectable_int import selectconcat as concat
from soc.decoder.orderedset import OrderedSet

class %s(ISACaller):

"""

class PyISAWriter(ISA):
    def __init__(self):
        ISA.__init__(self)

    def write_pysource(self, pagename):
        instrs = isa.page[pagename]
        isadir = get_isasrc_dir()
        fname = os.path.join(isadir, "%s.py" % pagename)
        with open(fname, "w") as f:
            iinf = ''
            f.write(header % pagename) # write out header
            # go through all instructions
            for page in instrs:
                d = self.instr[page]
                print (fname, d.opcode)
                pcode = '\n'.join(d.pcode) + '\n'
                print (pcode)
                pycode, rused = convert_to_python(pcode)
                # create list of arguments to call
                regs = list(rused['read_regs']) + list(rused['uninit_regs'])
                args = create_args(regs, 'self')
                # create list of arguments to return
                retargs = create_args(rused['write_regs'])
                # write out function.  pre-pend "op_" because some instrs are
                # also python keywords (cmp).  also replace "." with "_"
                op_fname ="op_%s" % page.replace(".", "_")
                f.write("    @inject(self.namespace)\n")
                f.write("    def %s(%s):\n" % (op_fname, args))
                pycode = pycode.split("\n")
                pycode = '\n'.join(map(lambda x: "        %s" % x, pycode))
                pycode = pycode.rstrip()
                f.write(pycode + '\n')
                if retargs:
                    f.write("        return (%s,)\n\n" % retargs)
                else:
                    f.write("\n")
                # accumulate the instruction info
                iinfo = "(%s, %s,\n                %s, %s)" % \
                            (op_fname, rused['read_regs'],
                            rused['uninit_regs'], rused['write_regs'])
                iinf += "    instrs['%s'] = %s\n" % (page, iinfo)
            # write out initialisation of info, for ISACaller to use
            f.write("    instrs = {}\n")
            f.write(iinf)

if __name__ == '__main__':
    isa = PyISAWriter()
    isa.write_pysource('fixedload')
    exit(0)
    isa.write_pysource('comparefixed')
    isa.write_pysource('fixedarith')
