from copy import deepcopy
import warnings
from Benchmarking.timeutils import Time as T
from Benchmarking.utils import LEN
from .Sensor import Sensor

class Timer(Sensor):

    def __init__(self, func_name, Class="Job", **kwargs):

        self.func_names = func_name
        self.Class = Class
        self.storage = {vn:[] for vn in self.func_names}


    def attach_to(self, *args, **kwargs):

        obj = args[0]
        
        for func_name in self.func_names: 
            timed_func = getattr(obj, func_name)

            if timed_func.__module__ == self.__module__:
                # already wrapped. avoid nested wrapping.
                continue 
            
            def make_wrapper(timed_f, f_name):
                def wrapper(*wargs, **kwwargs):

                    try:
                        start_time = T.now()
                        output = timed_f(*wargs, **kwwargs)  # Run the original forward pass                    

                    except Exception as e:
                        self.tele.error(str(e), "Cause:",  str(e.__cause__))
                        self.exception = str(e)
                        output = None

                    finally:
                        end_time = T.now()
                        #if self.time_to_record(i_episode):
                        self.collect(f_name, start_time, end_time)

                    return output
                return wrapper
            
            setattr(obj, func_name, make_wrapper(timed_func, func_name))  


    def collect(self, func_name, start_time, end_time):
        
        data = dict(start_time = start_time, 
                    end_time = end_time, 
                    duration = end_time - start_time )
        self.storage[func_name].append(deepcopy(data))


    def results(self, **kwargs):
        """Returns the harvested data in the form of a dictionary where 
            keys are variable names and values are the collected data
            
        Args:
            'kwargs' (not implemented) allows to filter the whole returned data by some of the collected variables
  
        Returns:
            _type_: _description_
        """
        return {fn:data for fn,data in self.storage.items() if fn in self.func_names}


    def reset(self):
        self.storage = {vn:[] for vn in self.func_names}

