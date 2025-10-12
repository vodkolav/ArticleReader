from datetime import datetime
import os

import json
from pathlib import Path
import pandas as pd
from Benchmarking.Pipeline import Pipeline
from Benchmarking.utils import span_grid, delaminate, recombine, read_json, write_json, filter_out_key
from Benchmarking.telemetry_manager import TelemetryManager
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO )

class Bench:

    def __init__(self, benchmarks_root = "benchmark", folder = None):

        if folder:
            # load from existing experiment folder 
            self.experiment_id = folder
        else:
            # create new experiment folder
            self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M")
        
        self.folder = os.path.join(benchmarks_root, self.experiment_id)
        # Ensure results directory exists
        os.makedirs(self.folder, exist_ok=True)

        self.benchmarks_root = benchmarks_root

        self.DONEcases = self.load_cases() 

        self.tele = TelemetryManager()


    def configure(self, pipeline: Pipeline):
        
        telemetry = TelemetryManager()
        pipeline.set_telemetry(telemetry)
        self.pipeline = pipeline
        # get template case from pipeline


    def check_grid(self, grid):
        if os.path.exists(os.path.join(self.folder, "grid.json")):
            exgrid = self.read_config("grid")
            if exgrid != grid:
                print("Warning: existing grid differs from new grid.")
                print("Existing grid:")
                print(exgrid)
                print("New grid:")
                print(grid)
        else:
            self.write_config(grid, filename = "grid", sort_keys=False)


    def unfurl_grid(self, case_template, grid):
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
        self.TODOcases = span_grid(grid, case_template)

        self.check_existing()

        #TODO: allow to add multiple grids for "or" combinations


    def check_existing(self):
        # check if some cases already done and filter them out of TODOcases
        if not self.DONEcases:
            return False
        doneconfigs = filter_out_key("summary", self.DONEcases)

        subm = len(self.TODOcases)

        self.TODOcases = [c for c in self.TODOcases if not filter_out_key("summary", c) in doneconfigs]

        logger.info("\nOut of %d submitted cases,\n  %d cases are already done.\n  %d are new and will be run. ", 
                    subm, len(doneconfigs), len(self.TODOcases))

        # write_json(self.TODOcases, "todo_cases.json", sort_keys=True)
        # write_json(self.DONEcases, "done_cases.json", sort_keys=True)
        # write_json(doneconfigs, "doneconfigs.json", sort_keys=True)

    def load_cases(self, patt = "*.coarse"):

        pth = Path(self.folder)
        
        paths = list(pth.glob(patt +".json"))
        
        if paths == []:
            print(f"No existing cases found in {self.folder} matching {patt}.")
            return []
        
        logger.info(f"loading {len(paths)} files from:", pth.absolute())
        logger.info(str(paths[0]), "...", sep = "\n")

        cases = []
        for i,p in enumerate(paths):
            cases.append(self.read_config(p.stem))
   
        return cases


    def summary(Cases):
        jn = pd.json_normalize(Cases)
        jnu = jn.nunique()
        cols = jnu.index[jnu > 1].tolist()
        return jn[cols]


    def read_config(self, filename):
        config_filepath = os.path.join(self.folder , f"{filename}.json")
        caSe = read_json(config_filepath)
        return caSe


    def write_config(self, caSe, filename, sort_keys = False):
        config_filepath = os.path.join(self.folder , f"{filename}.json")
        write_json(caSe, config_filepath, sort_keys=sort_keys, mode = 'x')


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
            experiment_run =self.pipeline.results()
            case_id = experiment_run['summary']["case_id"]
            coarse_data, fine_data = delaminate(experiment_run, self.pipeline.delamination_spec)
            self.write_config(coarse_data, f"{case_id}.coarse")
            self.write_config(fine_data, f"{case_id}.fine")

            #TODO: should be optional
            self.test_recombination(experiment_run, coarse_data, fine_data, case_id)

        run_results = self.tele.results()
        self.write_config(run_results, "run_results")

    def test_recombination(self, original, coarse_data, fine_data, case_id):

        recombined = recombine(coarse_data, fine_data, self.pipeline.delamination_spec)
        if  recombined == original:
            logger.info(f"Recombination successful for case {case_id}: Output matches original data!")
        else:
            logger.error(f"Recombination failed for case {case_id}: Output does NOT match original data!")
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

