from copy import deepcopy
import warnings
import numpy as np
from Benchmarking.timeutils import Time as T

class Harvester:
    """Collects variable values across iterations of a loop, and concatenates them at the end."""

    def __init__(self, varnames="", elems=None, on_size_mismatch = 'error', **kwargs):
        # TODO: add option to specify how to treat error when variable is not found in fnlocals, e.g ignore, warn, or raise error
        self.varnames = varnames
        self.elems = elems
        self.on_size_mismatch = on_size_mismatch
        self.storage = {vn:[] for vn in varnames}


    def get(self, fnlocals, vn):
        """if var name contains a dot, it is assumed to be a var.attribute.
        Extract this attribute

        Args:
            fnlocals (_type_): _description_
            vn (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if ("." in vn):
            pth = vn.split(".")
            if len(pth) > 2:
                raise ValueError("attributes of depth more than 2 not supported yet")    
            var, atr = pth
            varbl = fnlocals[var].__getattribute__(atr)
            return varbl
        else:
            return fnlocals[vn]



    def attach_to(self, dump_func, summary_func):

        def wrapper(*args, **kwargs):

            try:
                # self.record["start_time"] = T.now()
                output = dump_func(*args, **kwargs)  # Run the original forward pass                    
                i_episode = summary_func(mode="index")

            except Exception as e:
                self.tele.print(str(e), "Cause:",  str(e.__cause__))
                self.exception = str(e)
                output = None

            finally:
                #if self.time_to_record(i_episode):
                self.collect(*args, **kwargs)
                
                    # self.record["end_time"] = T.now()
                    # self.record["i"] = i_episode
                    # data = summary_func(mode = "data")
                    # self.record.update(data)
                    # self.episodes.append(deepcopy(self.record))
                    # self.record = {}

            return output
        return wrapper


    def collect(self, fnlocals):

        #TODO: validate that all variables that we want to collect are of the same size. 
        # Or size 1 - for scalar metrics such as circumference at step, etc.
        size = 1 
        scalars = {}
        for vn in self.varnames:

            varbl = self.get(fnlocals,vn)

            # if var is scalar - put it aside to later add as repeated
            # TODO: reconsider how we store scalars. 
            if np.isscalar(varbl):
                scalars[vn] = varbl
            else:
                # Extract specific elements if needed
                if self.elems is not None:
                    elems = fnlocals[self.elems]
                    content = varbl[elems]
                else:
                    content = varbl

                size = self.validate_size(size, content)
                
                self.storage[vn].append(deepcopy(content))
        
        for vn, content in scalars.items():
            self.storage[vn].append(np.repeat(content, size))
            # TODO: make scalars repeating happen only when 
            # returning results and optional - under repeat_scalars flag 

    def validate_size(self, size, content):
        if self.on_size_mismatch == 'ignore':
            return size
        
        thissize = len(content)
        if size!=1: 
            if thissize != size: 
                if self.on_size_mismatch == 'warn':
                    warnings.warn("Mismatch in collected variables sizes") 
                elif self.on_size_mismatch == 'error':
                    raise ValueError("All variables within particular collect() call must be either of the same size or scalars ")
        else: 
            size = thissize

        return size


    def add_once(self, **kwargs):
        for k,v in kwargs.items():
            if k in self.storage.keys():
                self.storage[k].append(deepcopy(v))
            else:
                self.storage[k] = [deepcopy(v)]


    def results(self, extract_attrs = True, break_2d_vectors = True,  **kwargs):
        """Returns the harvested data in the form of a dictionary where 
            keys are variable names and values are the collected data, 
            concatenated along individual collect()ions 
            
            Sometimes I want to harvest only some attributes of a variable, for ex. self.X 
            'extract_attrs' returns their name as just "X", without the "self." part
            
            Some variables are stored as 2D arrays of size [n,2] for x,y. 
            'break_2d_vectors' breaks them into separate x and y arrays so that they are easily read by pd.DataFrame()
            
            'kwargs' allows to filter the whole returned data by some of the collected variables
        Args:
            extract_attrs (bool, optional): rename var.attribute data to just attribute. Defaults to True.
            break_2d_vectors: whether to break 2d vectors. 
        Returns:
            _type_: _description_
        """

        e = lambda k: k.split(".")[-1] if extract_attrs else k

        store = {e(vn): np.concatenate(vals,axis=0) for vn, vals in self.storage.items()}

        if break_2d_vectors:
            tmp = [self.break_2D_vec(k,v) for k,v in store.items()]
            store = {k: v for d in tmp for k, v in d.items()}

        if kwargs:
            # kwargs = {"S":(5,6)}
            for k,v in kwargs.items():
                #TODO: Support multiple conditions with and/or
                mask = np.isin(store[k], v)
                outpt = {vn: vl[mask] for vn,vl in store.items()} 
            return outpt
        
        return store


    def sizes(self, results):

        lens = [len(v) for v in results.values()]
        allSameSize = all([a == lens[0] for a in lens])
        if not allSameSize: 
            if self.on_size_mismatch == 'warn':
                warnings.warn("Mismatch in collected variables sizes:" + str(lens)) 
            elif self.on_size_mismatch == 'error':
                raise ValueError("Mismatch in collected variables sizes:"
                                  + str(lens) 
                                  + ". make sure al variables within particular collect() "
                                  "call are either of the same size or scalars ")
        else:
            return lens[0]



    def summarize(self):

        results = self.results()

        res = {
            #TODO: decide if we need these
            # "sampling_type": self._sampling_type, # probably this is type "on_demand"
            # "sampling_value": self.sampling_value,
            # "extract_attrs" = ?, 
            # "break_2d_vectors" = ? 
            "varnames": self.varnames ,
            "on_size_mismatch": self.on_size_mismatch,
            "data": results,
            "summary": {
                "n_samples": self.sizes(results),
                    }
        }
        if self.elems:
            res['elems'] = self.elems 

        return res


    def last_step(self, extract_attrs = True, break_2d_vectors = True):

        e = lambda k: k.split(".")[-1] if extract_attrs else k

        store = {e(vn): vals[-1] for vn, vals in self.storage.items()}

        if break_2d_vectors:
            tmp = [self.break_2D_vec(k,v) for k,v in store.items()]
            store = {k: v for d in tmp for k, v in d.items()}

        return store


    def break_2D_vec(self, k, XY):
        """Breaks 2D vectors into 2 1D vectors.
           If it's already 1d, returned as is

        Args:
            k (name): name of vector
            XY (np.array): the vector

        Returns:
            dict: dict with 1 or 2 broken 1d vectors
        """
        if len(k)>1:
            nx, ny = [k[0],k[1]]
        else:
            nx, ny = [k + "x", k + "y"]

        if len(XY.shape)>1:
            X,Y = np.split(XY,2,axis=1)
            return {nx: X.squeeze(), ny: Y.squeeze()}
        else:
            return {k: XY}
    
