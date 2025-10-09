from datetime import datetime
import os

import json
from pathlib import Path
import pandas as pd
from Benchmarking.Pipeline import Pipeline
from Benchmarking.utils import span_grid, delaminate, recombine
from Benchmarking.telemetry_manager import TelemetryManager

class Bench:

    def __init__(self, benchmarks_root = "benchmark", folder = None):

        if folder:
            self.experiment_id = folder
        else:
            self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M")
        
        self.folder = os.path.join(benchmarks_root, self.experiment_id)
        # Ensure results directory exists
        os.makedirs(self.folder, exist_ok=True)

        self.benchmarks_root = benchmarks_root

        patt = "*.json"

        self.donecases = self.load_benchmarks(patt) 
       
        self.pathspec = [".tracks.episodes.data", 
            ".tracks.resources.data", 
            ".tracks.log.data"] 


    def configure(self, pipeline: Pipeline):
        
        telemetry = TelemetryManager()
        pipeline.set_telemetry(telemetry)
        self.pipeline = pipeline
        # get template case from pipeline


    def check_grid(self, grid):
        existing = os.path.join(self.folder, "grid.json")
        if os.path.exists(existing):
            exgrid = self.read_config(existing)
            if exgrid != grid:
                print("Warning: existing grid differs from new grid.")
                print("Existing grid:")
                print(exgrid)
                print("New grid:")
                print(grid)
        else:
            self.write_config(grid, filename = "grid", sort_keys=False)


    def unfurl_grid(self, case_template, grid, pathspec):
        # span grid to experiment_configs atop template case
        """
        grid = {'meta.chunk_length': [75, 100],
                'meta.batch_size': (2, 3),
                'model_tts.name': ['tts-tacotron2-ljspeech'],
                'model_voc.name': ['tts-hifigan-ljspeech'],
                'meta.device': ['CPU']
            }
        """
        self.check_grid(grid)
        self.pathspec = pathspec
        self.TODOcases = span_grid(grid, case_template)

        # TODO: check if some cases already done 
        #    if case not yet exists
        # if self.force or not (pd.DataFrame([self.case]).iloc[0] == self.donecases).all(axis=1).any():                                
        #     experiment_configs.append(case)
        # else:
        #     # if case in donecases and not force: skip and log
        #     print("data for case already exists:\n", self.case)

        #TODO: allow to add multiple grids for "or" combinations

    def load_benchmarks(self, patt = "*"):
        paths = Path(self.benchmarks_root).glob(patt +".json")
        experiments = [pd.read_json(p, orient="records") for p in paths]
        if experiments:
            experiments = pd.concat(experiments)
        return experiments


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


    def write_config(self, caSe, filename, sort_keys = False):
        config_filepath = os.path.join(self.folder , f"{filename}.json")
        """Writes experiment configurations to a JSON file"""
        try:
            with open(config_filepath, 'w+') as f:
                json.dump(caSe, f, indent=2, sort_keys=sort_keys)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_filepath}")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {config_filepath}")
            return


    def run_experiments(self, force = False):
        # sequentially
        # init the pipeline

        for i, config in enumerate(self.TODOcases):
            config['summary']["experiment_id"] = self.experiment_id
            status = self.pipeline.execute(config)
            if status != "Ok":
                print("fatal error in run_case. aborting")
                return
                # TODO: make it graceful

            print("saving benchmark data")
            experiment_run =self.pipeline.tele.results()
            case_sign = experiment_run['summary']["case_signature"]
            tstp    = experiment_run['summary']["timestamp"]

            case_id = tstp +"."+ case_sign
            experiment_run['summary']["case_id"] = case_id

            coarse_data, fine_data = delaminate(experiment_run, self.pathspec)
            self.write_config(coarse_data, f"{case_id}.coarse")
            self.write_config(fine_data, f"{case_id}.fine")

            self.test_recombination(experiment_run, coarse_data, fine_data, case_id)


    def test_recombination(self, original, coarse_data, fine_data, case_id):

        recombined = recombine(coarse_data, fine_data, self.pathspec)
        assert recombined == original

        self.write_config(original, f"{case_id}.original", sort_keys = True)
        self.write_config(recombined, f"{case_id}.recombined", sort_keys= True)


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
        self.benchmarks_root = self.benchmarks_root + "/" + datetime.now().strftime("%Y%m%d-%H%M")
        
        # Ensure results directory exists
        os.makedirs(self.benchmarks_root, exist_ok=True)
        
        # Create a multiprocessing Pool
        # The 'with' statement ensures the pool is properly closed
        all_results = []
        with multiprocessing.Pool(processes=num_cores) as pool:
            # pool.apply_async submits a single task and returns an AsyncResult object immediately.
            # This allows you to submit all tasks without waiting for each one to finish.
            async_results = []
            for i, config in enumerate(experiment_configs):
                print(f"Submitting experiment {i+1}/{len(experiment_configs)}: {config.get('name', 'unnamed')}")
                result = pool.apply_async(run_case, (config,self.benchmarks_root))
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

        return all_results, self.benchmarks_root




    def load_results(results_dir, patt = "*"):
        #TODO: check if thats a redundant function
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

