def wb_get(dut):
    """simulator process for getting memory load requests
    """
    mem = dut.mem
    wb = dut.cmpi.wb_bus()

    while not dut.stop:
        while True: # wait for dc_valid
            if dut.stop:
                return
            cyc = yield (wb.cyc)
            stb = yield (wb.stb)
            if cyc and stb:
                break
            yield
        addr = (yield wb.adr) << 3
        if addr not in mem:
            print ("    WB LOOKUP NO entry @ %x, returning zero" % (addr))

        # read or write?
        we = (yield wb.we)
        if we:
            store = (yield wb.dat_w)
            sel = (yield wb.sel)
            data = mem.get(addr, 0)
            # note we assume 8-bit sel, here
            res = 0
            for i in range(8):
                mask = 0xff << (i*8)
                if sel & (1<<i):
                    res |= store & mask
                else:
                    res |= data & mask
            mem[addr] = res
            print ("    DCACHE set %x mask %x data %x" % (addr, sel, res))
        else:
            data = mem.get(addr, 0)
            yield wb.dat_r.eq(data)
            print ("    DCACHE get %x data %x" % (addr, data))

        yield wb.ack.eq(1)
        yield
        yield wb.ack.eq(0)
        yield
