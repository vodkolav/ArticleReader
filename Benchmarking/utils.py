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


def classify(node, chld):
    if node == "":
        return '', None, "root"
    if "[" in node:
        sp = node.split('[')
        field = sp[0]
        key = sp[1][:-1]
        return field, key, "array"
    elif chld is None:
        return node, None, "leaf"
    else:
        return node, None, "dict"


def build_fine_query(doop, t="    ", pref = ""):
    res = []
   
    for k,v in doop.items():
        field, key, typ = classify(k, v)

        match typ:
            case "array":           
                key = f"{key}: .{key}"
                arg = build_fine_query(v, t+t)
                mem = f'(\n{t}{pref}.{field} // [] | map({{\n{t}{key},\n{t} {arg}\n{t}}})\n{t})'
                mem = f"{t}{field}: {mem}"

            case "leaf":
                mem = f"{t}{field}: {pref}.{field}"

            case "dict":
                arg = build_fine_query(v, t+t, f"{pref}.{field}")
                mem = f'{{\n{t+t}{arg}\n{t}}}'
                mem = f"{t}{field}: {mem}"

            case 'root':
                arg = build_fine_query(v, t, f"{pref}")
                mem = f'{{\n{arg}\n}}'

        res.append(mem)
    res = ",\n".join(res)
    
    return  res


def delaminate(original_json: Dict[str, Any], path_specs: Union[str, List[str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    deep = {}
    for p in path_specs:
        k = p.split('.')
        #print(p)
        deep = deep_dict(deep, k)

    fine_query = build_fine_query(deep)
    print(f"Fine Query: \n{fine_query}")

    coarse_query = ", ".join([re.sub(r"\[.*?\]", "[]", path) for path in path_specs])
    coarse_query = f"del({coarse_query})"
    print(f"Coarse Query: \n{coarse_query}")

    fine_data = jq.compile(fine_query).input(original_json).first()    
    coarse_data = jq.compile(coarse_query).input(original_json).first()

    return coarse_data, fine_data


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

        field, key, typ = classify(path, chld)

        if coarse_data.get(key) != fine_data.get(key) :
            raise ValueError(f"Key mismatch: {coarse_data.get(key)} != {coarse_data.get(key)}")

        match typ:
            case "array":
                if field in coarse_data:
                    coarse_data[field] = arr_merge(coarse_data[field], fine_data[field], key, chld)
                else:
                    if fine_data[field] != []:
                        coarse_data[field] = fine_data[field]

            case "leaf":
                coarse_data[field] = fine_data[field]

            case "dict":
                if field in coarse_data:
                    coarse_data[field] = deep_merge(coarse_data[field], fine_data[field], chld)

            case 'root':
                coarse_data = deep_merge(coarse_data, fine_data, chld)

    return coarse_data


def recombine(coarse_data: Dict[str, Any], fine_data: Dict[str, Any], path_specs: List[str] = None) -> Dict[str, Any]:
    
    deep = {}
    for p in path_specs:
        k = p.split('.')
        #print(p)
        deep = deep_dict(deep, k)

    combined = deep_merge(coarse_data, fine_data, deep)

    return combined


def permutations( grid):

    keys = list(grid.keys())
    n = len(keys)

    layers = [[{keys[l]:v} for v in grid[keys[l]]]  for l in range(n)]

    signatures = [None]

    res = [None]
    for i in range(0,n):
        l1 = res
        l2 = layers[i]
        res, signatures = mg(l1, l2, signatures)
    return res, signatures

cidp = ".summary.case_signature"


def get_path(pth, templ):
    val = jq.compile(f'{pth}?').input(templ).first()
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


def filter_out_key(key, templ):
    jqq = f'del(..| .{key}?)'
    return jq.compile(jqq).input(templ).first() 


def upd_path(pth, templ, val, force = False):
    
    if force or jq.compile(f'{pth}?').input(templ).first():
        val = qua(val)
        jqquery = f'{pth} = {val}'
        templ = jq.compile(jqquery).input(templ).first() 
        #print(f"set {k} to {o}")
        return templ
    else:
        print(f"key {pth} not in template")
        #TODO: check if we really need this case.
        # naturally, an update function should create missing paths. or not? 
        raise KeyError(pth)
    

def span_grid(grid, templ):
    # signature is the set of parameters 
    # and their values that uniquely identify this 
    # case from all the other cases in the grid
    cases = []
    cells, signatures = permutations(grid)

    for caSe, siGn in zip(cells, signatures):
        t = templ.copy()
        caSe[cidp] = signature_fmt(siGn)
        # brpt_anchr(k, 'meta.chunk_length')
        for k,v in caSe.items():
            t = upd_path(k, t, v, force=True)
        cases.append(t) 
    return cases


def brpt_anchr(var, val):
    if var == val:
        print("break me fully")


def signature_fmt(sig):
    if sig is None:
        return "single.case"
    res= [str(k).split(".")[-1] + "=" + str(v) for k,v in sig.items()]
    res = ".".join(res)
    return res


def mgn(x,y):
    # merge with none
    if x is None:
        if y is None:
            return None
        else:
            return y
    elif y is None:
        return x
    else:   
        return {**x, **y}

def mg(A,B, sign):
    res = [mgn(x, y) for x in A for y in B]
    if len(B) >1 :
        sign = [mgn(x, y) for x in sign for y in B]
    return res, sign


def read_json(filepath):
    """Reads experiment configurations from a JSON file"""
    try:
        with open(filepath, 'r') as f:
            caSe = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        return
    return caSe


def write_json(caSe, filepath, sort_keys = False, mode = 'w'):
    """Writes experiment configurations to a JSON file"""
    try:
        with open(filepath, mode) as f:
            json.dump(caSe, f, indent=2, sort_keys=sort_keys)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        return