#v2.py
import json
import jq
from typing import Dict, Any, List, Union, Tuple

import re 
# JSON delamination and recombination utility functions

def deep_dict(d, k):

    if k[0] in d:
        if len(k) > 1: 
            nd = d[k[0]]
            n = deep_dict(nd,k[1:]) 
        else:
            n = k[0]
        d[k[0]] = n
    else:
        # if k[0]==3:
        #     print("found 3")
        if len(k) > 1: 
            nd = {}
            n = deep_dict(nd,k[1:]) 
        else:
            n = None
        d[k[0]] = n
    return d



def build_fine_query(doop, t="    "):
    res = []
    chl = ""
    for k,v in doop.items():
        sp = k.split('[')
        field = sp[0]
        if len(sp) > 1:           
           
            key = sp[1][:-1]
            key = f"{key}: .{key}"
            arg = build_fine_query(v, t+t)
            mem = f'.{field} // [] | map({{\n{t}{key}, {arg}}})'

            #field = f"{field}: .{chl},"
        else:
            mem = "."+sp[0]
        
        res.append(f"{field}: ({mem})")
    res = ",\n".join(res)
    
    return  res



def delaminate(original_json: Dict[str, Any], path_specs: Union[str, List[str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    deep = {}
    for p in path_specs:
        k = p.split('.')[1:]
        #print(p)
        deep = deep_dict(deep, k)    

    root = list(deep.keys())[0]
    fix = root.split('[')[0]

    fine_query = build_fine_query(deep)
    fine_query = fine_query.replace(f"{fix}: (.{fix} // [] |", f".{fix} |=")[:-1]
    print(f"Fine Query: \n{fine_query}")

    coarse_query = ", ".join([re.sub(r"\[.*?\]", "[]?", path) for path in path_specs])
    coarse_query = f"del({coarse_query})"
    print(f"Coarse Query: \n{coarse_query}")


    fine_data = jq.compile(fine_query).input(original_json).first()    
    coarse_data = jq.compile(coarse_query).input(original_json).first()

    return coarse_data, fine_data


def splt(value):
    sp = value.split('[')
    field = sp[0]
    if len(sp) > 1:           
        key = sp[1][:-1]
    else: 
        key = None
    return field, key


def arr_merge(coarse, fine, key, deep_map):
    if len(coarse) == len(fine):
        # naive merge

        # res = []
        # for c,f in zip(coarse, fine):
        #     res.append(deep_merge(c, f, deep_map))
        try:
            res = [deep_merge(c, f, deep_map)  for c,f in zip(coarse, fine) ]
        except ValueError as e:
            raise NotImplementedError(f"Error merging arrays with key '{key}': {e}")
            #print("more involved merge needed, with syncing by id")

    else :
        raise ValueError("Coarse and fine arrays must have the same length for naive merge.")         
    return res


def deep_merge(coarse_data: Dict[str, Any], fine_data: Dict[str, Any], deep_map: Dict[str, str]) -> Dict[str, Any]:
    
    for path, chld in deep_map.items():
        field, key = splt(path)

        if coarse_data.get(key) != fine_data.get(key) :
            raise ValueError(f"Key mismatch: {coarse_data.get(key)} != {coarse_data.get(key)}")
        
        if key != None :
            if field in coarse_data:
                coarse_data[field] = arr_merge(coarse_data[field], fine_data[field], key, chld)
            else:
                if fine_data[field] != []:
                    coarse_data[field] = fine_data[field]
        else:
            coarse_data[field] = fine_data[field]
    return coarse_data


def recombine(coarse_data: Dict[str, Any], fine_data: Dict[str, Any], path_specs: List[str] = None) -> Dict[str, Any]:
    
    deep = {}
    for p in path_specs:
        k = p.split('.')[1:]
        #print(p)
        deep = deep_dict(deep, k)

    combined = deep_merge(coarse_data, fine_data, deep)

    return combined


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

def permutations( grid):

    keys = list(grid.keys())
    n = len(keys)

    layers = [[{keys[l]:v} for v in grid[keys[l]]]  for l in range(n)]

    def combine(prev, this):
        # print(prev)
        # print(this)
        tmp = [ t.copy() for t in this]
        [th.update(prev) for th in tmp]
        return tmp

    res = layers[0]
    for i in range(1,n):
        l1 = res
        l2 = layers[i]
        res = [combine(l, l2) for l in l1]
        res = sum(res,[])
    return res


def get_path(pth, templ):
    val = jq.compile(f'.{pth}?').input(templ).first()
    if val:
        return val
    else:
        raise KeyError(pth)


def qua(val):
    if isinstance(val, str):
        val = val.strip('"')
        val = f'"{val}"'
    elif isinstance(val, dict):
        val = json.dumps(val)
    return val


def upd_path(pth, templ, val):
    if jq.compile(f'.{pth}?').input(templ).first():
        val = qua(val)
        jqquery = f'.{pth} = {val}'
        templ = jq.compile(jqquery).input(templ).first() 
        #print(f"set {k} to {o}")
        return templ
    else:
        print(f"key {pth} not in template")
        raise KeyError(pth)
    

def span_grid(grid, templ):
    cases = []
    for caSe in permutations(grid):
        t = templ.copy()
        # brpt_anchr(k, 'meta.chunk_length')
        for k,v in caSe.items():
            t = upd_path(k, t, v)
        cases.append(t) 
    return cases

def brpt_anchr(var, val):
    if var == val:
        print("break me fully")