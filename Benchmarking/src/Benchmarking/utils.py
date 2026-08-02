#v2.py
import json
import jq
import orjson
import collections.abc
from typing import Dict, Any, List, Union, Tuple
import numpy as np
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
    #print(f"Fine Query: \n{fine_query}")

    coarse_query = ", ".join([re.sub(r"\[.*?\]", "[]", path) for path in path_specs])
    coarse_query = f"del({coarse_query})"
    #print(f"Coarse Query: \n{coarse_query}")

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
    # TODO: add option inplace=False, where output is new object and coarse_data remains the same
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

#TODO: put this in the right place
cidp = ".ID.case_signature"


def get_path(path, jsobj):
    k = path[0]
    if isinstance(k,str) and "[" in k:
        k,b = k.split("[")
        b = int(b.split("]")[0])
        path = [k,b]

    if len(path) > 1:
        v = jsobj[k]
        c = path[1:]
        return get_path(c,v) 
    else: 
        v = jsobj[k]
        return v



def qua(val):
    if val == [None,None]:
        #handle empty ranges, such as chunks_limit
        #TODO:this is ugly, make it more robust
        val = '[null,null]'
    elif isinstance(val, str):
        val = val.strip('"')
        val = f'"{val}"'
    elif isinstance(val, dict):
        val = orjson.dumps(val, option=20).decode('UTF-8')
    return val


def filter_out_keys(templ, *args):
    #deletes all occurences of key(s) from the templ recursively.
    #careful with it
    keys = [f".{key}?" for key in args]
    keys = ", ".join(keys)
    jqq = f'del(..| {keys})'
    return jq.compile(jqq).input(templ).first() 


def upd_path(path, host, guest):

    path = path.strip(".").split(".")

    guest = bury(path,guest)

    return merge_dicts(guest, host)


def bury(where, what):
    if where:
        k = where[0]
        if "[" in k:
            #TODO: implement
            raise ValueError("...list[2]... paths are not supported in 'where' yet. only 'dict.dict.dict...' paths are supported for now.")
        v = where[1:]
        r = {k:bury(v,what)}
        return r
    else: 
        return what



def merge_dicts(d, u):

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_dicts(d.get(k, {}), v)
        else:
            d[k] = v
    return d


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
            t = upd_path(k, t, v)
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


def write_json(caSe: dict, filepath, sort_keys = False, mode = 'w'):
    """Writes experiment configurations to a JSON file"""
    try:
        with open(filepath, mode) as f:
            json.dump(caSe, f, indent=2, sort_keys=sort_keys, cls=NumpyEncoder)
            # TODO: make "indent" variable to let user decide 
            # whether to output indented json for ease of human reading
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found at {filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        return


def isDebugging():
       import sys
       gettrace = getattr(sys, 'gettrace', None)

       if gettrace is None:
              print('No sys.gettrace')
              return False
       elif gettrace():
              print('Hmm, Big Debugger is watching me')
              return True
       else:
              print("Running in NOdebug mode")
              return False


def nunpack(a,n):
    """unpack 'a' into exactly n variables. 
    If a has less than n variables, assign None to the extra ones.
    
    a = "b.c.d."
    b,c,d,e,f,g = nunpack(a.split("."),6)
    b,c,d,e,f,g
    >>> ('b', 'c', 'd', '', None, None)

    Args:
        a (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    aa = a + [None]*n
    return aa[:n] 


def describe(subst):
    summ = dig({".":subst},-1)
    print(summ)


def shape(value):
    t = type(value).__name__
    if hasattr(value, '__getitem__'):    #some collection... 
        if hasattr(value, 'shape'):         # some numpy, can't dive in. or can? 
            l, s, r =   ("[", value.shape ,"]") 
        else:
            if hasattr(value, '__len__'):   # some builtin collection
                s = len(value)
                if t == "dict":                 # dict, can dive in
                    l, r = ("{", "}")
                elif t == "tuple":              # tuple, can dive in
                    l, r = ("(", ")")
                elif t == "list":               # list, can dive in
                    l, r = ("[", "]")
                else:                           # probably str, can't dive in 
                    l, r = ("(", ")")
            else:
                raise ValueError("Scary, very scary, we don't know what that is. If we knew wat that is, we don't know what that is. ")
    else:
        l, s, r = ("","","")  # scalar, can't dive in; show value

    return t, l, s, r


def one(k,v,f):
    
    t, l, s, r = shape(v) 
    dug = dig(v,f)
    c =  dug # "\n".join(dug)
    i = "    "*f
    fmt = f"{i}{k}: {t}{l}{s}{r} = {c}"
    return fmt


def dig(subst, f):               
    if isinstance(subst, dict):
        summ = ["  "] + [ one(f'\"{k}\"',v,f+1) for k,v in subst.items() ]
        summ = "\n".join(summ)
    elif isinstance(subst, list) :
        lastitem = f'[{len(subst)}]' 
        summ = "\n" + one(lastitem,subst[-1],f+1)
    elif isinstance(subst, np.ndarray) :
        lastitem = f'[{len(subst)}]' 
        summ = "\n" + one(lastitem,subst[-1],f+1)
    elif isinstance(subst, str):
        summ = f"'{subst.replace("\n","")}'"
        # summ = f"'{subst.replace("\n","\n" + "    "*(f+1))}'"
    else:
        summ = str(subst)
    return  summ


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types 
        taken from https://stackoverflow.com/a/49677241/7097017
    """    

    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)