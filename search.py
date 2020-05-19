
import datetime as dt
import json

#filename = 'test'
filename = 'coral-qd1a.190821.007-factory-e915f51a.zip'
#patternfile = 'patterns'
patternsfile = 'BinaryPatterns/patterns.json'


def readPatterns(patternsfile):

    with open(patternsfile, "r") as f:
        texts = json.load(f)

    #print(texts)

    patterns = list()

    longest = 0
    for (k,v) in texts.items():
        longest = max(longest, len(k))
        patterns.append(bytearray.fromhex(k))
    return (patterns,longest)

(patterns,longest) = readPatterns(patternsfile)


starttime = dt.datetime.now()
print('begin')

results = dict()

with open(filename, "rb") as f:
    buffer = b'\x00' + f.read(11)
    byte = f.read(1)
    offset = 0
    while byte and offset < 1e6:
        buffer = buffer[1:] + byte
        for pattern in patterns:
            l = len(pattern)
            if buffer[:l] == pattern:
                hexoffset = '%02x'%offset
                strpat = bytearray.hex(pattern) 
                if strpat in results :
                    res = results[strpat]
                    res.append(hexoffset)
                    results[strpat] = res
                else:
                    results[strpat] = [hexoffset]
        offset +=1
        byte = f.read(1)

print(results)
print('done in ' , dt.datetime.now() - starttime )





