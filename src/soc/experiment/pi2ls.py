"""
    PortInterface                LoadStoreUnitInterface

    is_ld_i/1                    x_ld_i
    is_st_i/1                    x_st_i

    data_len/4                   x_mask/16  (translate using LenExpand)

    busy_o/1                     most likely to be x_busy_o
    go_die_i/1                   rst?
    addr.data/48                 x_addr_i[4:] (x_addr_i[:4] goes into LenExpand)
    addr.ok/1                    probably x_valid_i & ~x_stall_i

    addr_ok_o/1                  no equivalent.  *might* work using x_stall_i
    addr_exc_o/2(?)              m_load_err_o and m_store_err_o

    ld.data/64                   m_ld_data_o
    ld.ok/1                      probably implicit, when x_busy drops low
    st.data/64                   x_st_data_i
    st.ok/1                      probably kinda redundant, set to x_st_i
"""
