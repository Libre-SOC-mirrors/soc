from collections import OrderedDict, namedtuple


class BitRange(OrderedDict):
    """BitRange: remaps from straight indices (0,1,2..) to bit numbers
    """
    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return list(self)[subscript]
        else:
            return self[subscript]

def decode_instructions(form):
    res = {}
    accum = []
    for l in form:
        if l.strip().startswith("Formats"):
            l = l.strip().split(":")[-1]
            l = l.replace(" ", "")
            l = l.split(",")
            for fmt in l:
                if fmt not in res:
                    res[fmt] = [accum[0]]
                else:
                    res[fmt].append(accum[0])
            accum = []
        else:
            accum.append(l.strip())
    return res

def decode_form_header(hdr):
    res = {}
    count = 0
    hdr = hdr.strip()
    print (hdr.split('|'))
    for f in hdr.split("|"):
        if not f:
            continue
        if f[0].isdigit():
            idx = int(f.strip().split(' ')[0])
            res[count] = idx
        count += len(f) + 1
    return res

def find_unique(d, key):
    if key not in d:
        return key
    idx = 1
    while "%s_%d" % (key, idx) in d:
        idx += 1
    return "%s_%d" % (key, idx)


def decode_line(header, line):
    line = line.strip()
    res = {}
    count = 0
    print ("line", line)
    prev_fieldname = None
    for f in line.split("|"):
        if not f:
            continue
        end = count + len(f) + 1
        fieldname = f.strip()
        if not fieldname or fieldname.startswith('/'):
            if prev_fieldname is not None:
                res[prev_fieldname] = (res[prev_fieldname], header[count])
                prev_fieldname = None
            count = end
            continue
        bitstart = header[count]
        if prev_fieldname is not None:
            res[prev_fieldname] = (res[prev_fieldname], bitstart)
        res[fieldname] = bitstart
        count = end
        prev_fieldname = fieldname
    res[prev_fieldname] = (bitstart, 32)
    return res


def decode_form(form):
    header = decode_form_header(form[0])
    res = []
    print ("header", header)
    for line in form[1:]:
        dec = decode_line(header, line)
        if dec:
            res.append(dec)
    fields = {}
    falternate = {}
    for l in res:
        for k, (start,end) in l.items():
            if k in fields:
                if (start, end) == fields[k]:
                    continue # already in and matching for this Form
                if k in falternate:
                    alternate = "%s_%d" % (k, falternate[k])
                    if (start, end) == fields[alternate]:
                        continue
                falternate[k] = fidx = falternate.get(k, 0) + 1
                fields["%s_%d" % (k, fidx)] = (start, end)
            else:
                fields[k] = (start, end)
    return fields


class DecodeFields:

    def __init__(self, bitkls=BitRange, bitargs=(), fname="fields.txt"):
        self.bitkls = bitkls
        self.bitargs = bitargs
        self.fname = fname

    def create_specs(self):
        self.forms, self.instrs = self.decode_fields()
        self.form_names = forms = self.instrs.keys()
        for form in forms:
            fields = self.instrs[form]
            fk = fields.keys()
            Fields = namedtuple("Fields", fk)
            instr = Fields(**fields)
            setattr(self, "Form%s" % form, instr)
        # now add in some commonly-used fields (should be done automatically)
        # note that these should only be ones which are the same on all Forms
        # note: these are from microwatt insn_helpers.vhdl
        self.RS = self.FormX.RS
        self.RT = self.FormX.RT
        self.RA = self.FormX.RA
        self.RB = self.FormX.RB
        self.SI = self.FormD.SI
        self.UI = self.FormD.UI
        self.L = self.FormD.L
        self.SH32 = self.FormM.SH
        self.MB32 = self.FormM.MB
        self.ME32 = self.FormM.ME
        self.LI = self.FormI.LI
        self.LK = self.FormI.LK
        self.AA = self.FormB.AA
        self.Rc = self.FormX.Rc
        self.OE = self.FormXO.Rc
        self.BD = self.FormB.BD
        self.BF = self.FormX.BF
        self.CR = self.FormXL.XO # used by further mcrf decoding
        self.BB = self.FormXL.BB
        self.BA = self.FormXL.BA
        self.BT = self.FormXL.BT
        self.FXM = self.FormXFX.FXM
        self.BO = self.FormXL.BO
        self.BI = self.FormXL.BI
        self.BH = self.FormXL.BH
        self.D = self.FormD.D
        self.DS = self.FormDS.DS
        self.TO = self.FormX.TO
        self.BC = self.FormA.BC
        self.SH = self.FormX.SH
        self.ME = self.FormM.ME
        self.MB = self.FormM.MB

    def decode_fields(self):
        with open(self.fname) as f:
            txt = f.readlines()
        forms = {}
        reading_data = False
        for l in txt:
            print ("line", l)
            l = l.strip()
            if len(l) == 0:
                continue
            if reading_data:
                if l[0] == '#':
                    reading_data = False
                else:
                    forms[heading].append(l)
            if not reading_data:
                assert l[0] == '#'
                heading = l[1:].strip()
                #if heading.startswith('1.6.28'): # skip instr fields for now
                    #break
                heading = heading.split(' ')[-1]
                print ("heading", heading)
                reading_data = True
                forms[heading] = []

        res = {}
        inst = {}

        for hdr, form in forms.items():
            print ("heading", hdr)
            if heading == 'Fields':
                i = decode_instructions(form)
                for form, field in i.items():
                    inst[form] = self.decode_instruction_fields(field)
            #else:
            #    res[hdr] = decode_form(form)
        return res, inst

    def decode_instruction_fields(self, fields):
        res = {}
        for field in fields:
            f, spec = field.strip().split(" ")
            d = self.bitkls(*self.bitargs)
            idx = 0
            for s in spec[1:-1].split(","):
                s = s.split(':')
                if len(s) == 1:
                    d[idx] = int(s[0])
                    idx += 1
                else:
                    start = int(s[0])
                    end = int(s[1])
                    while start <= end:
                        d[idx] = start
                        idx += 1
                        start += 1
            f = f.replace(",", "_")
            unique = find_unique(res, f)
            res[unique] = d

        return res

if __name__ == '__main__':
    dec = DecodeFields()
    dec.create_specs()
    forms, instrs = dec.forms, dec.instrs
    for hdr, form in forms.items():
        print ()
        print (hdr)
        for k, v in form.items():
            #print ("line", l)
            #for k, v in l.items():
            print ("%s: %d-%d" % (k, v[0], v[1]))
    for form, field in instrs.items():
        print ()
        print (form)
        for f, vals in field.items():
            print ("    ", f, vals)
    print (dec.FormX)
    print (dec.FormX.A)
    print (dir(dec.FormX))
    print (dec.FormX._fields)
