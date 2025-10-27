import os

import json
from pathlib import Path
import pandas as pd
from Benchmarking.Pipeline import Pipeline
from Benchmarking.utils import span_grid, delaminate, recombine, read_json, write_json, filter_out_keys
from Benchmarking.timeutils import Time as T
from Benchmarking.telemetry_manager import TelemetryManager
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO )

class Bench:

#TODO: support multiple runs of an experiment?

    def __init__(self, benchmarks_root = "benchmark",
                       output_root = "output", 
                       folder = None):
        
        self.TELE = TelemetryManager()       
    
        self.output_root = output_root
        self.benchmarks_root = benchmarks_root

        if folder:
            # load from existing experiment folder 
            self.experiment_id = folder
            self.folder = os.path.join(benchmarks_root, self.experiment_id)
            # check if folder exists and has experiment.json            
            if os.path.exists(os.path.join(self.folder, "experiment.json")):
            # load existing experiment 
                self.config = self.read_config("experiment")
            else:
                self.config = self.config_template() 

                self.TELE.warning(f"experiment.json not found in {self.folder}, starting fresh.")
                #.write_config(grid, filename = "grid", sort_keys=False)
            
            self.DONEcases = self.load_cases() 
            # 
        else:
            # create new experiment folder
            s = os.path.sep
            self.experiment_id = T.timestamp(fmt=f"%Y%m%d{s}%H%M")
            self.folder = os.path.join(benchmarks_root, self.experiment_id)
            # Ensure results directory exists
            self.TELE.print(f" creating new experiment in {self.folder}.")
            os.makedirs(self.folder, exist_ok=True)
            self.config = self.config_template()
            self.DONEcases = []
        
        self.TELE.start(self.config)

        self.delamination = False
        self.test_recombination = False


    def config_template(self):
        #TODO: write down system parameters?
        return {'summary': {"case_signature":"experiment", "case_index":0}}


    def configure(self, pipeline: Pipeline):
        
        telemetry = TelemetryManager()
        pipeline.set_telemetry(telemetry)
        self.pipeline = pipeline
        # get template case from pipeline

    @property
    def grid(self):
        return self.config["grid"] if "grid" in self.config else {}

    @grid.setter
    def grid(self, val):
        if self.grid and (self.grid != val):
            msg = f"""
            Warning: existing grid differs from new grid.
            Existing grid:
            {json.dumps(self.grid)}
            New grid:
            {json.dumps(val)}
            """
            self.TELE.warning(msg)
        self.config["grid"] = val 


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
        self.grid = grid
        self.TODOcases = span_grid(grid, case_template)
        self.TELE.print(f"Unfurled grid into {len(self.TODOcases)} TODOcases")

        self.check_existing()

        #TODO: allow to add multiple grids for "or" combinations


    def check_existing(self):
        # check if some cases already done and filter them out of TODOcases
        if not self.DONEcases:
            return False
        doneconfigs = filter_out_keys(self.DONEcases, "summary", "tracks")

        subm = len(self.TODOcases)

        self.TODOcases = [c for c in self.TODOcases if not filter_out_keys(c, "summary", "tracks") in doneconfigs]

        self.TELE.print(f"\nOut of {subm} submitted cases,\n  {len(doneconfigs)} cases are already done.\n  {len(self.TODOcases)} are new and will be run. ")

        # write_json(self.TODOcases, "todo_cases.json", sort_keys=True)
        # write_json(self.DONEcases, "done_cases.json", sort_keys=True)
        # write_json(doneconfigs, "doneconfigs.json", sort_keys=True)


    def load_cases(self, patt = "2025*"):

        pth = Path(self.folder)
        
        paths = list(pth.glob(patt +".json"))
        
        if paths == []:
            self.TELE.print(f"No existing cases found in {self.folder} matching {patt}.")
            return []
        
        self.TELE.print(f"loading {len(paths)} files from:" + str( pth.absolute()))
        #self.TELE.print(str(paths[0]), "...", sep = "\n")

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


    def write_config(self, caSe, filename, sort_keys = False, mode = 'x'):
        config_filepath = os.path.join(self.folder , f"{filename}.json")
        write_json(caSe, config_filepath, sort_keys=sort_keys, mode = mode)


    def addresil(self, new_case):
        if 'tracks' in new_case:
            new_case['tracks']['resources']['resilient'] = "dbg/" 


    def stamp_case(self,i,config):
        # assigns all the ids and timestamps to the case
        # those depend on time of run of the case
        run_epoch = T.now()

        tstp = T.timestamp(run_epoch)

        case_sign = config['summary']["case_signature"]

        case_id = tstp +"."+ case_sign

        summary = {
            "experiment_id": self.experiment_id,
            "case_id": case_id,
            "case_index": i, 
            "output_root": self.output_root,
            "start_time": run_epoch,
            "timestamp": tstp,
        }

        config['summary'].update(summary)
        pass


    def run_experiments(self, force = False):
        # sequentially
        # init the pipeline

        for i, config in enumerate(self.TODOcases):
            self.addresil(config)
            self.stamp_case(i, config)
            status = self.pipeline.execute(config)
            
            experiment_run =self.pipeline.results()
            self.save_case(experiment_run)
            
            #dump the TELE of the bench to disk after every case - 
            #otherwise if run fails at some case, the whole bench log is lost
            self.TELE.collect_log()
            self.write_config(self.TELE.results(), "experiment", mode='w+')

            if status == "Fatal":
                self.TELE.error("fatal error in run_case. aborting run")
                return
                # TODO: make it graceful
                # TODO: move from TODOcases to DONEcases            

        self.TELE.print("Benchmark run complete!")
        self.TELE.end()       
        self.write_config(self.TELE.results(), "experiment", mode='w+')


    def save_case(self, experiment_run):
        case_id = experiment_run['summary']["case_id"]
        self.TELE.print(f"saving benchmark data to: {case_id}.json")
        if self.delamination:

            coarse_data, fine_data = delaminate(experiment_run, self.pipeline.delamination_spec)
            self.write_config(coarse_data, f"{case_id}.coarse")
            self.write_config(fine_data, f"{case_id}.fine")

            if self.test_recombination:
                self.test_recomb(experiment_run, coarse_data, fine_data, case_id)
        else: 
            self.write_config(experiment_run, f"{case_id}")


    def test_recomb(self, original, coarse_data, fine_data, case_id):

        recombined = recombine(coarse_data, fine_data, self.pipeline.delamination_spec)
        if  recombined == original:
            self.TELE.print(f"Recombination successful for case {case_id}: Output matches original data!")
        else:
            self.TELE.error(f"Recombination failed for case {case_id}: Output does NOT match original data!")
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
                self.TELE.warning("Could not detect CPU count, defaulting to 1 core.")
                num_cores = 1
            else:
                self.TELE.print(f"Detected {num_cores} CPU cores. Using {num_cores} workers.")

        # Separate every run of battery of tests to its own dir
        self.benchmarks_root = self.benchmarks_root + "/" + T.now().strftime("%Y%m%d-%H%M")
        
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
                self.TELE.print(f"Submitting experiment {i+1}/{len(experiment_configs)}: {config.get('name', 'unnamed')}")
                result = pool.apply_async(run_case, (config,self.benchmarks_root))
                async_results.append(result)

            # Wait for all tasks to complete and collect results
            self.TELE.print("\nWaiting for experiments to complete...")
            for i, res in enumerate(async_results):
                try:
                    # .get() will block until the result is ready
                    # You can add a timeout if you want to handle unresponsive processes
                    experiment_result = res.get()
                    all_results.append(experiment_result)
                    self.TELE.print(f"Experiment {i+1}/{len(experiment_configs)}")
                except Exception as e:
                    self.TELE.error(f"Error running experiment {i+1}: {e}")
                    all_results.append({"error": str(e), "config": experiment_configs[i]})

        self.TELE.print("\nAll experiments finished.")
        self.TELE.print("\n--- Summary of Results ---")
        for res in all_results:
            if "error" in res:
                self.TELE.error(f"  FAILED: {res['config'].get('name', 'Unnamed')} - Error: {res['error']}")
            else:
                self.TELE.print(res["status"], res["timestamp"])

        return all_results, self.benchmarks_root

