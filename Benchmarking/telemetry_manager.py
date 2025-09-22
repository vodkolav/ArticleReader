# telemetry/telemetry_manager.py
from collections import defaultdict
import numpy as np
import json
from datetime import datetime
import time

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
    """
    Manages the collection of training and evaluation metrics.
    """
    def __init__(self, experiment_config: dict ):
        
        self.metrics_data = experiment_config

        self.summary = {
            "init_rss_mb": self.get_memory_usage_mb()
        }

        #TODO: define float format, ex: Avg Reward (last 100): {avg_reward:.2f}

        # self.env = experiment_config["env"]

        #self.algorithm = experiment_config["algorithm"]
        
        #self.strategy = experiment_config["strategy"]

        self.tot_episodes = 0

        limit = self.meta.get("telemetry_episodes_limit", 100)
        self.samplePoints = list(range(limit))

        # Use defaultdicts to store lists of metrics per episode/step
        self.episodes = []
        #self.algorithm_specific_metrics = dict(list) # For things like TD error, policy change

        #self.reset_episode(0)

    @property
    def total_episodes(self):
        return self.tot_episodes

    @total_episodes.setter
    def total_episodes(self, value):
        limit = len(self.samplePoints)
        if value < limit:
            limit = value
        
        self.samplePoints = np.int64(np.linspace(0,value, limit))
        invl = np.round(value/limit, decimals=2)
        print("tracking and reporting once every", invl, "episodes")
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

    def record_episode(self, data: dict):
        """Records metrics at the end of an episode."""
        if self.i_episode in self.samplePoints:
            self.episodes.append(deepcopy(data))


    def report(self, what, newline = False):
        if newline:
            print(what)
        else:
            print(f"\r{what}" , end='')


    def start(self, num_episodes):
        # Start telemetry reporting for an experiment
        self.total_episodes = num_episodes
        run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.summary["start_time"] = run_timestamp
        self.summary["pid"] = os.getpid()

        self.summary["experiment_id"] = self.metrics_data["meta"]["name"] + "_" +\
                    str(run_timestamp) + "_" + str(os.getpid())
        # alg_name = self.algorithm["name"]
        # self.report(f" Starting {alg_name} over {num_episodes} episodes", newline=True)

    def end(self):
        # Optional: Print end message
        end_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.summary["end_time"] = end_timestamp
        # alg_name = self.algorithm["name"]
        self.report(f"\n {self.meta["name"]} ended. Total episodes recorded: {len(self.episodes)}", newline=True)

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