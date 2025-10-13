# telemetry/telemetry_manager.py
from collections import defaultdict
import numpy as np
import json
from datetime import datetime
import time

from Benchmarking.EpisodeTracker import EpisodeTracker
from Benchmarking.MemoryMonitor import MemoryMonitor
import Benchmarking.utils as butils
import os
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
        self._print(what,"info")


    def _print(self, what, type = "info"):
        #TODO: change this
        entry = {
            "type": type,
            "time": self.now(),
            "message": what,
        }
        self.log.append(entry)
        self.report(self.dt_format(entry))


    def warning(self, what):
        self._print(what,"warning")


    def error(self, what):
        self._print(what,"error")


    def dt_format(self, entry: dict):
        entry['time'] = datetime.fromtimestamp(entry['time'])\
                                .strftime(self.tstp_format)
        disp = "[{type}] {time}: {message}"
        return disp.format(**entry)


    def timestamp(self, entry = None):
        if entry:
            return datetime.fromtimestamp(entry)\
                           .strftime(self.tstp_format) 
        else:
            return self.timestamp(self.now())


    def report(self, what, newline = False) -> None:
        if newline:
            print(what)
        else:
            print(f"\r{what}" , end='')


    def start(self, new_case):
        # Start telemetry reporting for an experiment
        self.case = new_case
        #self.summary = self.case.get("summary", {})



    # def collect_episodes(self):
    #     self.case["tracks"]["episodes"]["data"] = self.episodes


    def collect_log(self):
        self.case["tracks"]["log"]["data"] = self.log


    def add_memory_monitor(self, func, config: dict, label):
        monitor = MemoryMonitor(**config)
        func = monitor.attach_to(func)
        self.sensors[label] = monitor
        return func


    def add_EpisodeTracker(self, ep_func, summ_func, config: dict, label):
        trckr = EpisodeTracker(**config)
        ep_func = trckr.attach_to(ep_func, summ_func)
        self.sensors[label] = trckr
        return ep_func


    def collect_sensors(self):

        for k,v in self.sensors.items():
            sens_summary = v.summarize()
            self.case = butils.upd_path(k, self.case, sens_summary)


    def end(self):
        # Optional: Print end message
        self.collect_sensors()
        #self.collect_episodes()        
        # alg_name = self.algorithm["name"]
        self.case["summary"] = self.summary
        #nm = self.summary["experiment_id"]
        #self.report(f"\n {nm} ended. Total episodes recorded: {len(self.episodes)}", newline=True)
        self.collect_log()


    def results(self):
        return self.case


    def progress(self, message = ""):
        # Optional: Print progress
        if self.i_episode in self.samplePoints:            
            msg = f"\r {self.mode} Episode {self.i_episode}/{self.total_episodes}" + message
            self.report(msg)

