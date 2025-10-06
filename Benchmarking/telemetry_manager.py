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

        self.summary = {}


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
        entry = {
            "type": "info",
            "time": self.now(),
            "message": what,
        }
        self.log.append(entry)

        self.report(self.dt_format(entry))


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


    def misc(self, data):
        self.case['misc'] = data



    def start(self, new_case):
        # Start telemetry reporting for an experiment
        self.case = new_case
        self.summary = self.case["summary"]
        #self.config_scheduling()

        run_epoch = self.now()
        self.summary["start_time"] = run_epoch
        self.summary["timestamp"] = self.timestamp(run_epoch)

        self.summary["init_rss_mb"] = self.sensors[".tracks.resources"].get_memory_usage_mb()


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
        #print("combining tts_profiler results")

         for k,v in self.sensors.items():
            mem_summary = v.summarize()
            self.case = butils.upd_path(k, self.case ,mem_summary)


    def end(self):
        # Optional: Print end message
        end_timestamp = self.now()
        self.summary["end_time"] = end_timestamp
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


    def save_metrics(self, filename: str):
        """Saves collected metrics to a JSON file."""
        # metrics_data = {
        #     "metadata": self.metadata,
        #     "env": self.env,
        #     "algorithm": self.algorithm,
        #     "strategy": self.strategy,             
        #     "episodes": self.episodes,  
        # }
        with open(filename, 'w') as f:
            json.dump(self.metrics_data, f, indent=4, cls=NumpyEncoder)
        print(f"Metrics saved to {filename}")


    def load_metrics(self, filename: str):
        """Loads metrics from a JSON file."""
        try:
            with open(filename, 'r') as f:
                self.metrics_data = json.load(f)
                # self.metadata = metrics_data.get("metadata", {}) 
                # self.episodes = metrics_data.get("episodes", [])
            print(f"Metrics loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Metrics file not found at {filename}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}")