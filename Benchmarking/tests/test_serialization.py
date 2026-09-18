

from Benchmarking.utils import read_json, write_json, describe



testfile = "Benchmarking/tests/data/case_np_orient.json"
roundtrip_file = "Benchmarking/tests/data/case_roundtrip.json"

HV = "horz"

caSe = read_json(testfile, np_orient = HV)
write_json(caSe, roundtrip_file, np_orient = HV)
after = read_json(roundtrip_file, np_orient = HV)
descafter = describe(after, how = "return")
desccaSe = describe(caSe, how = "return")
assert desccaSe == descafter 


HV = "vert"

caSe = read_json(testfile, np_orient = HV)
write_json(caSe, roundtrip_file, np_orient = HV)
after = read_json(roundtrip_file, np_orient = HV)
descafter = describe(after, how = "return")
desccaSe = describe(caSe, how = "return")
assert desccaSe == descafter 