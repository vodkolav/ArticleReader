#v2.py

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


def recombine(coarse_data: Dict[str, Any], fine_data: Dict[str, Any], path: List[str] = None) -> Dict[str, Any]:
    
    
    return "not implemented yet"

