# telemetry/telemetry_manager.py
from collections import defaultdict
import numpy as np
import json
from datetime import datetime
import time

from Benchmarking.EpisodeTracker import EpisodeTracker
from Benchmarking.MemoryMonitor import MemoryMonitor
from Benchmarking.MemoryProfiler import MemoryProfiler
import Benchmarking.utils as butils
import os

import logging
from copy import deepcopy
#from algorithms.agent import RLAgent

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types 
        taken from https://stackoverflow.com/a/49677241/7097017
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class TelemetryManager:
    #TODO: rename to just Telemetry
    #TODO: implement proper logging
    """
    Manages the collection of training and evaluation metrics.
    """
    def __init__(self): # experiment_config: dict ):
        
        #self.metrics_data = experiment_config

        # sensors are independent components that 
        # usually track system resources, such as memory/GPU etc.
        # run on separate threads
        # not in sync with telemetry episodes/epochs
        self.sensors = {}

        self.case = {}

        self.log = []

        #self.summary = {}


        #TODO: define float format, ex: Avg Reward (last 100): {avg_reward:.2f}

        #self.tstp_format = "%Y%m%d-%H%M%S-%f"
        self.tstp_format = "%Y%m%d-%H%M%S"

        # self.env = experiment_config["env"]

        #self.algorithm = experiment_config["algorithm"]
        
        #self.strategy = experiment_config["strategy"]
        # if we start counting samples from 0, then 
        # the one before it is -1
        self.last_sample = -1 
        self.tot_episodes = 0

        #limit = self.meta.get("telemetry_episodes_limit", 100)
        limit = 100
        self.samplePoints = list(range(limit))

        # Use defaultdicts to store lists of metrics per episode/step
        
        #self.algorithm_specific_metrics = dict(list) # For things like TD error, policy change

        #self.reset_episode(0)


    def intercept_logs(self, LIBRARY_LOGGER_NAME = 'third_party_lib' ):
        # 1. Get the logger instance for the third-party library
        #    Replace 'third_party_lib' with the actual name of the library's logger
        
        library_logger = logging.getLogger(LIBRARY_LOGGER_NAME)

        # 2. Set the desired logging level 
        #    The logger will only process events *at or above* this level (e.g., INFO, DEBUG)
        library_logger.setLevel(logging.INFO) 

       
        # 3. Create an instance of your custom handler
        custom_handler = TelemetryManagerHandler(self)

        # Optional: Add a Formatter
        # This ensures the log message passed to your manager is formatted correctly
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        custom_handler.setFormatter(formatter)

        # 4. Attach the custom handler to the library's logger
        library_logger.addHandler(custom_handler)

        # 5. Optional: Stop propagation to the root logger
        #    If the logs are still showing up on the console (because they're being handled 
        #    by the default 'root' logger), you can stop them from reaching it.
        library_logger.propagate = False 


    @property
    def total_episodes(self):
        return self.tot_episodes

    @total_episodes.setter
    def total_episodes(self, value):


        #TODO: implement different forms of scheduling reports
        # - total episodes to report (requires how many total episodes will be)
        # - once every x episodes (frequency)
        # - time-based
        # - on demand: whenever something happens (log)
        self.tot_episodes = value

    @property
    def output_root(self):
        return self.case['summary']['output_root']


    @property
    def mode(self):
        return "Training" if self._current_episode["is_training"] else "Evaluating"



    # def reset_episode(self, i_episode):
    #     """Resets metrics for a new episode."""
    #     self.i_episode = i_episode
    #     self._current_episode = { 
    #         "i": i_episode,             
    #         "reward": 0,
    #         "length": 0,
    #         "replay": [],
    #         "info": [],
    #         "internal_state":{}, 
    #         "start": time.time()
    #     } 

    #def record_step(self, reward: float, info: dict = None, frame=None):


        #self._current_episode["info"].append(info if info is not None else {})
        # You can record other step-specific info if needed from the 'info' dict


    def now(self):
        # TODO: variable format
        return time.time()


    def print(self, what):
        #TODO: rename to info
        self._print(what,"info")


    def Log(self, entry:dict): 
        self.log.append(entry)
        self.display(self.format_entry(entry), newline=True)


    def _print(self, what, type = 'info', **kwargs):
        #TODO: change this

        time = kwargs["time"] if 'time' in kwargs else self.now()
        entry = {
            "type": type,
            "time": time,
            "message": what,
            }
        self.Log(entry)


    def warning(self, what):
        self._print(what, "warning")


    def error(self, what):
        self._print(what, type = "error")


    def debug(self, what):
        self._print(what, type = "debug")


    def format_entry(self, entry: dict):
        newentry = entry.copy()
        newentry['timestamp'] = self.timestamp(newentry['time'])
        disp = "[{type}] {timestamp}: {message}"
        return disp.format(**newentry)


    def timestamp(self, entry = None):
        if entry:
            return datetime.fromtimestamp(entry)\
                           .strftime(self.tstp_format) 
        else:
            return self.timestamp(self.now())


    def display(self, what, newline = False) -> None:
        if newline:
            print(what)
        else:
            print(f"\r{what}" , end='')


    def start(self, new_case):
        # Start telemetry reporting for an experiment
        #self.summary = self.case.get("summary", {})

        if self.case == new_case:
            self.warning("all fields are already identical, which should not happen")  # raise Error?;  
        
        force = False
        if self.case == {}:
            self.case = new_case
            force = True  # first run, so all initializers must run


        self.case['summary'] = new_case['summary']

        run_epoch = self.now()
        self.case['summary']["start_time"] = run_epoch

        tstp = self.timestamp(run_epoch)
        self.case['summary']["timestamp"] = tstp

        case_sign = new_case['summary']["case_signature"]

        case_id = tstp +"."+ case_sign
        self.case['summary']["case_id"] = case_id

        return force


    def case_filename(self):
        case_id = self.case['summary']["case_id"]
        experiment_id = self.case['summary']["experiment_id"]
        exp_dir = os.path.join(self.output_root, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        pth = os.path.join(exp_dir, case_id)
        return pth


    # def collect_episodes(self):
    #     self.case["tracks"]["episodes"]["data"] = self.episodes


    def collect_log(self):
        self.case = butils.upd_path(".tracks.log.data", self.case, [], force=True)
        # FIXME: fix this ugly hack
        self.case["tracks"]["log"]["data"] = self.log        


    def add_memory_monitor(self, func, config: dict, label):
        monitor = MemoryMonitor(**config)
        func = monitor.attach_to(func)
        self.sensors[label] = monitor
        return func


    def AttachSensor(self, obj, func_name, label, **kwargs):
        tracks = ".tracks."
        level, sensor = label.split(tracks)
        #item = "data"
        func = getattr(obj, func_name)
        pth = f"{level}{tracks}{sensor}"

        match sensor:
            case "episodes": 
                config = butils.get_path(pth, self.case)
                snsr = EpisodeTracker(**config)
                summ_func = getattr(obj, kwargs['summary_func'])
                func = snsr.attach_to(func, summ_func)

            case "resources": 
                
                if "config" in kwargs:
                    config = kwargs["config"]
                else:
                    config = butils.get_path(pth, self.case, default= {})

                snsr = MemoryMonitor(**config)
                #summ_func = kwargs['summary_func']
                func = snsr.attach_to(func)

            case "profile": 
                if  label in self.sensors:
                    snsr = self.sensors[label]
                else:                        
                    if "config" in kwargs:
                        config = kwargs["config"]
                    else:
                        config = butils.get_path(pth, self.case, default= {})
                    snsr = MemoryProfiler(**config)
                func = snsr.attach_to(func)

        setattr(obj,func_name, func)
        snsr.tele = self
        self.sensors[label] = snsr        

            #case "log":  TODO: decide if me make it a sensor too. or leave it special

    def add_EpisodeTracker(self, ep_func, summ_func, config: dict, label):
        trckr = EpisodeTracker(**config)
        ep_func = trckr.attach_to(ep_func, summ_func)
        self.sensors[label] = trckr
        return ep_func


    def collect_sensors(self):

        for k,v in self.sensors.items():            
            sens_summary = deepcopy(v.summarize())            
            if "profile" in k and sens_summary['data']!=[]:
                self.save_other(sens_summary)

            self.case = butils.upd_path(k, self.case, sens_summary, force=True)


    def save_other(self,v):
        #TODO:temporary hack, shold be intergrated into resilient monitor
        data = v["data"]
        cid = self.case["summary"]["case_id"]
        pth = f"dbg/{cid}"
        os.makedirs(pth, exist_ok=True)      

        for prof in data:
            id = prof["id"]
            fnm = prof["function_name"]
            with open(pth + f"/{fnm}_{id}.prof", "w+") as fl:
                fl.write(prof["profile_log"])

    def end(self):
        # Optional: Print end message
        self.collect_sensors()
        #self.collect_episodes()        
        # alg_name = self.algorithm["name"]
        #nm = self.summary["experiment_id"]
        #self.report(f"\n {nm} ended. Total episodes recorded: {len(self.episodes)}", newline=True)
        self.collect_log()
        self.log = []
        # create a report
        self.case['summary']["end_time"] = self.now()


    def results(self):
        return deepcopy(self.case)


    def progress(self, message = ""):
        # TODO implement progression percentage of individual cases
        # Optional: Print progress
        if self.i_episode in self.samplePoints:            
            msg = f"\r {self.mode} Episode {self.i_episode}/{self.total_episodes}" + message
            self.display(msg)


class TelemetryManagerHandler(logging.Handler):
    """
    A custom handler that sends log records to a specified manager object.
    """
    def __init__(self, manager_object: TelemetryManager):
        super().__init__()
        self.manager = manager_object

    def emit(self, record: logging.LogRecord):
        """
        Called by the logging system for each log record.
        """

        # Format the record before passing it to the manager
        #message =  self.format(record)

        entry = {
            "message" : record.message,
            "type" : record.levelname.lower(),
            "time" : record.created,
            "source": record.name # 'speechbrain.utils.fetching'
        }
        # other properties from record can be added

        # Pass the formatted message and/or the raw record to your manager
        # The exact method call depends on your manager's API
        self.manager.Log(entry)

