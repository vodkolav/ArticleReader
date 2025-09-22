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



    def run_experiments(self, processed_text, grid, force = False):
        """
        grid = {"chunk_length": range(50, 500, 50),
                "batch_size": (1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 200),
                "tts_model": ["tts-tacotron2-ljspeech"],
                "vocoder_model": ["tts-hifigan-ljspeech"],
                "device": ["CPU"], 
            }
        grid
        """        
        self.provider = "speechbrain"
        
        self.case_objects = {}
        self.case = {}

        for d in grid["device"]:
            self.init_device(d)

            for tts_model_name in grid["tts_model"]:                
                self.init_tts_model(tts_model_name)

                for voc_model_name in grid["vocoder_model"]:     
                    self.init_voc_model(voc_model_name)

                    for chunk_length in grid["chunk_length"]:
                        self.init_chunker(processed_text, chunk_length)

                        for batch_size in grid["batch_size"]:
                            self.init_batch(batch_size)

                            print("-"*30)
                            #   if case not yet exists
                            if force or not (pd.DataFrame([self.case]).iloc[0] == self.donecases).all(axis=1).any():                                
                                experiment_run = self.run_case()
                                print("saving benchmark data")
                                with open("benchmark/" + experiment_run[0]["experiment_id"] + ".json", "w+") as f:
                                    json.dump(experiment_run,f, indent=4)
                            else:
                                print("data for case already exists:\n", self.case)
        print("experiment complete.")


    def make_case(C: Constants, algo_name ,alpha, gamma, lambda_, epsilon = ("linear", 1) , theta = 1e-5, ):
        
        decay, eps = epsilon

        # descr = {"case": i, "algo_name": algo_name , "alpha": alpha, "gamma": gamma, 
        #          "lambda_":lambda_, "epsilon": epsilon, "theta": theta}

        Case =  {
            "metadata": {
                "name": f"",
                "description": f"Experiment with {algo_name} algorithm, gamma={gamma}, lambda={lambda_}",
                "num_training_episodes": C.NUM_TRAINING_EPISODES,
                "num_eval_episodes": C.NUM_EVAL_EPISODES,
                "render_evaluation": C.RENDER_EVALUATION,
                "save_ansi_frames": False,
                "telemetry_episodes_limit": 256,
                "skip": C.SKIP
            },
            "env": {
                "name": C.ENV_ID,
            } ,
            "algorithm": {
                "name": algo_name ,
                "params": {
                    "alpha": alpha,  
                    "gamma": gamma,
                    "lambda_": lambda_,
                    "theta": theta,  # Only for Dynamic Programming
                }
            },
            "strategy": {
                "name": "EpsilonGreedy",
                "params":{
                    "decay": decay,
                    "initial_epsilon": eps,
                    "min_epsilon": 0.01,
                    "epsilon_decay_episodes": C.NUM_TRAINING_EPISODES
                }
            }
        }
        return Case

    def summary(Cases):
        jn = pd.json_normalize(Cases)
        jnu = jn.nunique()
        cols = jnu.index[jnu > 1].tolist()
        return jn[cols]




    def read_configs(config_filepath):
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



    def run_battery_of_experiments(experiment_configs: list, num_cores: int = None, results_dir="results"):
        """
        Reads experiment configurations from a JSON file and runs them in parallel.

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
        results_dir = results_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M")
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a multiprocessing Pool
        # The 'with' statement ensures the pool is properly closed
        all_results = []
        with multiprocessing.Pool(processes=num_cores) as pool:
            # pool.apply_async submits a single task and returns an AsyncResult object immediately.
            # This allows you to submit all tasks without waiting for each one to finish.
            async_results = []
            for i, config in enumerate(experiment_configs):
                print(f"Submitting experiment {i+1}/{len(experiment_configs)}: {config.get('name', 'unnamed')}")
                result = pool.apply_async(run_case, (config,results_dir))
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

        return all_results, results_dir


    def load_experiment(data):
        meta = data["metadata"]

        # Flatten the algorithm parameters into the metadata
        # I'll deal with strategy parameters later
        algo = data["algorithm"]
        algo.update(algo["params"])
        algo.pop("params", None)
        meta.update(algo)

        strat = data["strategy"]
        meta.update({"decay": strat["params"]["decay"],
                    "initial_epsilon": strat["params"]["initial_epsilon"],})

        episodes = pd.DataFrame(data["episodes"])
        episodes["exp_id"] = meta["id"]

        return meta, episodes


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




                                   
