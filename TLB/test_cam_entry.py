from nmigen.compat.sim import run_simulation

from CamEntry import CamEntry
 
# This function allows for the easy setting of values to the Cam Entry
# unless the key is incorrect
# Arguments:
#   dut: The CamEntry being tested
#   c (command): NA (0), Read (1), Write (2), Reserve (3)
#   k (key): The key to be set
#   d (data): The data to be set  
def set_cam_entry(dut, c, k, d):
    # Write desired values
    yield dut.command.eq(c)
    yield dut.key_in.eq(k)
    yield dut.data_in.eq(d)
    yield
    # Reset all lines
    yield dut.command.eq(0)
    yield dut.key_in.eq(0)
    yield dut.data_in.eq(0)    
    yield
    
# Verifies the given values via the requested operation
# Arguments:
#   p (Prefix): Appended to the front of the assert statement
#   e (Expected): The expected value
#   o (Output): The output result
#   op (Operation): (0 => ==), (1 => !=)
def check(p, o, e, op):
    if(op == 0):
        assert o == e, p + " Output " + str(o) + " Expected " + str(e)
    else:
        assert o != e, p + " Output " + str(o) + " Not Expecting " + str(e)     

# Checks the key state of the CAM entry
# Arguments:
#   dut: The CamEntry being tested
#   k (Key): The expected key
#   op (Operation): (0 => ==), (1 => !=)
def check_key(dut, k, op):
    out_k = yield dut.key
    check("Key", out_k, k, op)   
   
# Checks the data state of the CAM entry
# Arguments:
#   dut: The CamEntry being tested
#   d (Data): The expected data
#   op (Operation): (0 => ==), (1 => !=)
def check_data(dut, d, op):
    out_d = yield dut.data
    check("Data", out_d, d, op)   
  
# Checks the match state of the CAM entry
# Arguments:
#   dut: The CamEntry being tested
#   m (Match): The expected match  
#   op (Operation): (0 => ==), (1 => !=)
def check_match(dut, m, op):
    out_m = yield dut.match
    check("Match", out_m, m, op)  
  
# Checks the state of the CAM entry
# Arguments:
#   dut: The CamEntry being tested
#   k (key): The expected key  
#   d (data): The expected data
#   m (match): The expected match  
#   k_op (Operation): The operation for the key assertion (0 => ==), (1 => !=)
#   d_op (Operation): The operation for the data assertion (0 => ==), (1 => !=)
#   m_op (Operation): The operation for the match assertion (0 => ==), (1 => !=)
def check_all(dut, k, d, m, k_op, d_op, m_op):
    yield from check_key(dut, k, k_op)
    yield from check_data(dut, d, d_op)
    yield from check_match(dut, m, m_op)
    
# This testbench goes through the paces of testing the CamEntry module
# It is done by writing and then reading various combinations of key/data pairs
# and reading the results with varying keys to verify the resulting stored
# data is correct.
def testbench(dut):
    # Check write
    command = 2
    key = 1
    data = 1
    match = 0
    yield from set_cam_entry(dut, command, key, data)
    yield from check_all(dut, key, data, match, 0, 0, 0)
    
    # Check read miss
    command = 1
    key = 2
    data = 1
    match = 0 
    yield from set_cam_entry(dut, command, key, data)
    yield from check_all(dut, key, data, match, 1, 0, 0) 
    
    # Check read hit
    command = 1
    key = 1
    data = 1
    match = 1
    yield from set_cam_entry(dut, command, key, data)
    yield from check_all(dut, key, data, match, 0, 0, 0) 
    
    # Check overwrite
    command = 2
    key = 2
    data = 5
    match = 0
    yield from set_cam_entry(dut, command, key, data)
    yield
    yield from check_all(dut, key, data, match, 0, 0, 0) 
    
    # Check read hit
    command = 1
    key = 2
    data = 5
    match = 1
    yield from set_cam_entry(dut, command, key, data)
    yield from check_all(dut, key, data, match, 0, 0, 0) 
    
    # Extra clock cycle for waveform
    yield
    
if __name__ == "__main__":
    dut = CamEntry(4, 4)
    run_simulation(dut, testbench(dut), vcd_name="Waveforms/cam_entry_test.vcd")
    print("CamEntry Unit Test Success")
