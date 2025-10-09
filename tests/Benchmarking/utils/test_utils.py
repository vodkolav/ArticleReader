#deep = {}

from Benchmarking.utils import build_fine_query, delaminate, recombine
import json

# simple example
# deep = {1: {2: {4 : None}}}

# deep = deep_dict(deep, [1,2,5])
# deep

# # print(json.dumps(deep, indent=2))
# # {3: {6: none}}

# deep = deep_dict(deep, [1,3,6])

# deep = deep_dict(deep, [1,2,7,8])
# deep

pathspec = [".BackupCfg[Id].cfg[ID].Paths", 
            ".BackupCfg[Id].progress.wins", 
            ".BackupCfg[Id].port",
            ".AnotherCfg.switches.data",
            ".AnotherCfg.points[i].x"] 


original_json_data = {
    "BackupCfg": [
    {
        "Id": "00",
        "type": "filesystem",
        "repository": "hurr",
        "url": "test.example.com",
        "port": "394",
        # this is for testing for absence. 
        # the result should have "progress":{"wins":[]} here
        # "progress":
        # {
        #     "id":5, 
        #     "name":"test", 
        #     "status":"ok", 
        #     # "wins":[100,200,300,400],
        #     }
        # ,
        "cfg": [
        {
            "Default": "true",
            "ID": "trunk00",
            "Paths": [
            "/etc",
            "/home",
            "/var",
            "/usr/local",
            "/opt",
            "/root"
            ],
            "Cron": "33 0 * * *"
        }
        ]
    },
    {
        "Id": "01",
        "port": "394",
        "type": "filesystem2",
        "repository": "burr",
        "url": "test.example.com",
        "cfg": [
        {
            "ID": "trunk01",
            "Paths": [
            "/opt/example",
            "/opt/var_example"
            ],
            "Cron": "*/30 0-23 * * *"
        }
        ],
        "progress":
            {
            "name":"test", 
            "status":"ok", 
            "wins":[100,200,300,400],            
            }

    }
    ],
    "AnotherCfg": {
        "version": "1.0",
        "description": "just a test",
        "points": [
            {
                "i": 0,
                "x": 1,
            }
        ],
        "switches": {
            "data": [1,2,3,4,5]
        }
    }
}

wd = 'tests/Benchmarking/utils/' # working directory

def dump(data, filename='data'):
    with open(wd + filename + ".json", 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

def load( filename='data'):
    with open(wd + filename + ".json", 'r') as f:
        return json.load(f)


print("--- Original JSON Data ---")
dump(original_json_data, 'original')

# test dynamic generation of fine query part
import jq 
from Benchmarking.utils import deep_dict

deep = {}
for p in pathspec:
    k = p.split('.')[1:]
    #print(p)
    deep = deep_dict(deep, k)  

fine_query = build_fine_query(deep)

fine_query = f"{{\n{fine_query}\n}}" # temporary fix for outer {}

with open(wd+"q_fine.jq", 'w') as f:
    f.write(fine_query)

target_fine_query = """
{
  BackupCfg: (
    .BackupCfg // [] | map({
      Id: .Id,
      cfg: (
        .cfg // [] | map({
            ID: .ID,
            Paths: .Paths
        })
        ),
        progress: {
        wins: .progress.wins
      },
      port: .port
    })
  ),
  AnotherCfg: {
    switches: {
      data: .AnotherCfg.switches.data
    },
    points: (
        .AnotherCfg.points // [] | map({
      i: .i,
      x: .x
    })
    )
  }
}
"""

with open(wd+"q_tgt.jq", 'w') as f:
    f.write(target_fine_query)

# bash:
# code --diff q_fine.jq q_tgt.jq 

# test if result of generated query matches target
fine_data = jq.compile(fine_query).input(original_json_data).first() 


# --- Test full delamination run ---

coarse_data, fine_data = delaminate(original_json_data, pathspec)

dump(fine_data, 'fine')
dump(coarse_data, 'coarse')


# bash:
# code --diff original.json test/fine.json 
# code --diff original.json test/coarse.json 


# --- Recombination 
try:
    recombined_output = recombine(coarse_data, fine_data, pathspec)
    dump(recombined_output, 'recombined')

    # Verify if recombined is identical to original
    print("\n--- Verification ---")
    if recombined_output == original_json_data:
        print("Recombination successful: Output matches original data!")
    else:
        print("Recombination failed: Output does NOT match original data!")

except ValueError as e:
    print(f"\nError during recombination: {e}")


# # --- Example of an ambiguous case (simulated) ---
# print("\n--- Testing Ambiguous ID Detection ---")
# ambiguous_coarse = {
#     "Data": [
#         {"id": "A", "name": "Item A", "tag": "X"},
#         {"id": "B", "name": "Item B", "tag": "Y"}
#     ]
# }
# ambiguous_fine = {
#     "Data": [
#         {"id": "A", "value": 100, "tag": "X"}, # 'tag' is also common
#         {"id": "B", "value": 200, "tag": "Y"}
#     ]
# }
# try:
#     print("\nAttempting merge with ambiguous IDs:")
#     recombine(ambiguous_coarse, ambiguous_fine)
# except ValueError as e:
#     print(f"Caught expected error: {e}")

# # --- Example of no common ID key (simulated) ---
# print("\n--- Testing No Common ID Key Detection ---")
# no_common_coarse = {
#     "Data": [
#         {"unique_id1": "A", "name": "Item A"},
#     ]
# }
# no_common_fine = {
#     "Data": [
#         {"unique_id2": "A", "value": 100},
#     ]
# }
# try:
#     print("\nAttempting merge with no common ID key:")
#     recombine(no_common_coarse, no_common_fine)
# except ValueError as e:
#     print(f"Caught expected error: {e}")

from Benchmarking.utils import permutations

def test_permutations():

    grid = {"A": [1,2,3,4,5,6],
            "B": "a b c d e f g h i j".split(' '),
            "C": ["U", "V"],
            "D": ["J","K"],
            "E": ["P"], 
            }
    res = permutations(grid)
    
    import json
    with open("check.json", 'w+') as f: 
        json.dump(res, f, indent=4)


    import pandas as pd 
    df = pd.read_json("check.json")
    print(df)

    print(len(df.drop_duplicates())) 