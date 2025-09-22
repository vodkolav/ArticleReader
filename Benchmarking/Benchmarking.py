from datetime import datetime
import os

import json
from pathlib import Path
import pandas as pd


class Bench:

    def __init__(self, benchmark_dir = "benchmark", patt = "*"):
        self.benchmark_dir = benchmark_dir
        self.donecases = self.load_benchmarks(patt)

    def load_benchmarks(self, patt = "*"):
        paths = Path(self.benchmark_dir).glob(patt +".json")
        experiments = [pd.read_json(p, orient="records") for p in paths]
        bnch_data = pd.concat(experiments)
        return bnch_data[["device", "tts_model", "vocoder_model", "chunk_length", "batch_size"]].copy()


    def summary(Cases):
        jn = pd.json_normalize(Cases)
        jnu = jn.nunique()
        cols = jnu.index[jnu > 1].tolist()
        return jn[cols]


    def read_configs(config_filepath):
        """Reads experiment configurations from a JSON file"""
        try:
            with open(config_filepath, 'r') as f:
                experiment_configs = json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_filepath}")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {config_filepath}")
            return

        print(f"Loaded {len(experiment_configs)} experiments from {config_filepath}")
        return experiment_configs




    def run_experiments(self, experiment_configs: list):
        
        self.benchmark_dir = self.benchmark_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M")
        
        # Ensure results directory exists
        os.makedirs(self.benchmark_dir, exist_ok=True)
                            #   if case not yet exists
        if force or not (pd.DataFrame([self.case]).iloc[0] == self.donecases).all(axis=1).any():                                
            experiment_run = self.run_case()
            print("saving benchmark data")
            with open("benchmark/" + experiment_run[0]["experiment_id"] + ".json", "w+") as f:
                json.dump(experiment_run,f, indent=4)
        else:
            print("data for case already exists:\n", self.case)


    def run_experiments_parallel(self, experiment_configs: list, num_cores: int = None ):
        """
        Runs experiment_configs in parallel.

        Args:
            config_filepath: Path to the JSON file containing experiment configurations.
            num_cores: Number of CPU cores to use. Defaults to all available cores.
        """

        if num_cores is None:
            num_cores = os.cpu_count()
            if num_cores is None:
                print("Warning: Could not detect CPU count, defaulting to 1 core.")
                num_cores = 1
            else:
                print(f"Detected {num_cores} CPU cores. Using {num_cores} workers.")

        # Separate every run of battery of tests to its own dir
        self.benchmark_dir = self.benchmark_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M")
        
        # Ensure results directory exists
        os.makedirs(self.benchmark_dir, exist_ok=True)
        
        # Create a multiprocessing Pool
        # The 'with' statement ensures the pool is properly closed
        all_results = []
        with multiprocessing.Pool(processes=num_cores) as pool:
            # pool.apply_async submits a single task and returns an AsyncResult object immediately.
            # This allows you to submit all tasks without waiting for each one to finish.
            async_results = []
            for i, config in enumerate(experiment_configs):
                print(f"Submitting experiment {i+1}/{len(experiment_configs)}: {config.get('name', 'unnamed')}")
                result = pool.apply_async(run_case, (config,self.benchmark_dir))
                async_results.append(result)

            # Wait for all tasks to complete and collect results
            print("\nWaiting for experiments to complete...")
            for i, res in enumerate(async_results):
                try:
                    # .get() will block until the result is ready
                    # You can add a timeout if you want to handle unresponsive processes
                    experiment_result = res.get()
                    all_results.append(experiment_result)
                    print(f"Experiment {i+1}/{len(experiment_configs)}")
                except Exception as e:
                    print(f"Error running experiment {i+1}: {e}")
                    all_results.append({"error": str(e), "config": experiment_configs[i]})

        print("\nAll experiments finished.")
        print("\n--- Summary of Results ---")
        for res in all_results:
            if "error" in res:
                print(f"  FAILED: {res['config'].get('name', 'Unnamed')} - Error: {res['error']}")
            else:
                print(res["status"], res["timestamp"])

        return all_results, self.benchmark_dir




    def load_results(results_dir, patt = "*"):
        pth = Path(results_dir)
        
        paths = list(pth.glob(patt +".json"))
        
        print(f"loading {len(paths)} files from:", pth.absolute())
        print(str(paths[0]), "...", sep = "\n")

        experiments = ['']*len(paths)
        episodes = []
        for i,p in enumerate(paths):
            # try:
                with open(p) as f:
                    data = json.load(f)
                    experiment, exp_episodes = load_experiment(data)
                    experiments[i] = experiment
                    episodes.append(exp_episodes)
            # except Exception as ex: 
            #     print("oops:", p)
        experiments = pd.DataFrame(experiments)
        episodes = pd.concat(episodes)
        print("done.")
        return experiments, episodes




                                   
