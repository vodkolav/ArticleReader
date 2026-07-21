# experiment/run_single_experiment.py

import os
from datetime import datetime
import json
from dataclasses import dataclass
import multiprocessing
import os

from pathlib import Path
import pandas as pd 


# Import your classes
#from Benchmarking.RLPileline import Experiment # Your Experiment class

@dataclass
class Constants:
    ENV_ID: str = "Taxi-v3"
    NUM_TRAINING_EPISODES: int = 1000
    NUM_EVAL_EPISODES: int = 100
    RENDER_EVALUATION: bool = False 
    SKIP: bool = False


if __name__ == '__main__':
    # It's crucial that code run by multiprocessing.Pool is either in another file
    # or inside the 'if __name__ == "__main__":' block itself.
    # The function run_single_experiment is imported, so it's safe.

    config_file = "configs/sarsa_battery.json"
    # Set num_cores to None to use all available CPU cores,
    # or a specific number like 4, 8, etc.
    results = run_battery_of_experiments(config_file, num_cores=None)

    # You can now further process or analyze 'results'
    # e.g., generate plots comparing performance, save summary to CSV, etc.