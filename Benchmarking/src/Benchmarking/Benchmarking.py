import os

import json
from pathlib import Path
# import pandas as pd
from Benchmarking.Pipeline import Pipeline
from Benchmarking.utils import span_grid, delaminate, recombine, read_json, write_json, filter_out_keys
from Benchmarking.utils import get_path, upd_path, describe
from Benchmarking.timeutils import Time as T
from Benchmarking.telemetry_manager import TelemetryManager
from Benchmarking.Case import Case
from Benchmarking.Job import Job
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO )


class Bench:

    #TODO: support multiple runs of an experiment?

    pipeline: Pipeline

    tele: TelemetryManager # telemetry for current case

    job: Job
    # CAse: Case

    TELE: TelemetryManager # telemetry for the whole experiment

    EXPERMT: Case

    i: int

    DONEcases: list = []

    TODOcases: list = []


    @property
    def bench_folder(self):
        return os.path.join(self.benchmarks_root, self.experiment_id)


    @property
    def output_folder(self):
        return os.path.join(self.output_root, self.experiment_id)


    def __init__(self, benchmarks_root = "benchmark",
                       output_root = "output", 
                       folder = None,
                       pattrn = "*",
                       onerror = "skip"):
        
        self.onerror = onerror
        self.TELE = TelemetryManager(self)
        # self.TELE.start(self.config)
    
        self.output_root = output_root
        self.benchmarks_root = benchmarks_root

        abspath = Path(benchmarks_root).resolve().as_posix()
        self.TELE.print("Absolute path: ", abspath)

        self._CAse: Case = {}

        if folder:
            # load from existing experiment folder 
            self.experiment_id = folder
            self.load_cases(pattrn)
            # 
        else:
            # create new experiment folder
            s = os.path.sep
            self.experiment_id = T.timestamp(fmt=f"%Y%m%d{s}%H%M")
            # Ensure results directory exists
            self.TELE.print(f" creating new experiment in {self.bench_folder}.")
            os.makedirs(self.bench_folder, exist_ok=True)
            os.makedirs(self.output_folder, exist_ok=True)
            self.EXPERMT = self.config_template()
            self.DONEcases = []
        

        self.delamination = False
        self.test_recombination = False


    def config_template(self):
        #TODO: write down system parameters?
        return Case({
            "config": {},
            'ID': {"case_signature":"experiment",
                   "case_index":0},
            "tracks":  {},
            "summary": {}})


    def configure(self, pipeline: Pipeline):
        

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
        self.TODOcases = [Case(c) for c in span_grid(grid, case_template)]
        self.TELE.print(f"Unfurled grid into {len(self.TODOcases)} TODOcases")

        self.check_existing()

        #TODO: allow to add multiple grids for "or" combinations


    def set_cases(self, cases: dict):
        """just add cases as is, without unfurling a grid

        Args:
            cases (list): list of dir, every dir is a case
        """
        #TODO: validate all incoming cases signatures are also unique among themselves.
        self.TODOcases = [Case(c) for c in cases]
        self.check_existing()


    def check_existing(self):
        # check if some cases already done and filter them out of TODOcases
        if not self.DONEcases:
            return False

        subm = len(self.TODOcases)

        self.TODOcases = [c for c in self.TODOcases if not c in self.DONEcases]

        self.TELE.print(f"\nOut of {subm} submitted cases,\n  {subm - len(self.TODOcases)} cases are already done.\n  {len(self.TODOcases)} are new and will be run. ")

        # write_json(self.TODOcases, "todo_cases.json", sort_keys=True)
        # write_json(self.DONEcases, "done_cases.json", sort_keys=True)
        # write_json(doneconfigs, "doneconfigs.json", sort_keys=True)


    def load_cases(self, patt = "*"):

        pth = Path(self.bench_folder)
        
        paths = pth.glob(patt +".json")
        
        if paths == []:
            self.TELE.print(f"No existing cases found in {self.bench_folder} matching {patt}.")
            return []
        
        #self.TELE.print(str(paths[0]), "...", sep = "\n")

        self.DONEcases = []
        for i,p in enumerate(paths):
            acase = Case(self.read_config(p.stem))
            # skip loading data of the whole experiment as an individual case
            if acase.case_signature == 'experiment':

                self.EXPERMT = acase
                self.TELE.print("loading experiment from ", str(p))
            else:
                self.DONEcases.append(acase)

        if not hasattr(self, "EXPERMT"):
            self.EXPERMT = self.config_template() 
            self.TELE.warning(f"experiment.json not found in {self.bench_folder}, starting fresh.")

   
        self.TELE.print(f"loaded {len(self.DONEcases)} files from:" + str( pth.absolute()))


    def summary(Cases):
        jn = pd.json_normalize(Cases)
        jnu = jn.nunique()
        cols = jnu.index[jnu > 1].tolist()
        return jn[cols]


    def read_config(self, filename):
        config_filepath = os.path.join(self.bench_folder , f"{filename}.json")
        caSe = read_json(config_filepath)
        return caSe


    def write_config(self, caSe: dict, filename, path = None, sort_keys = False, mode = 'x'):
        path = path if path else self.bench_folder 

        if self.delamination:

            coarse_data, fine_data = delaminate(caSe, self.pipeline.delamination_spec)

            write_json(coarse_data, 
                       os.path.join(path, f"{filename}.coarse.json"),
                       sort_keys=sort_keys, mode = mode)

            write_json(fine_data, 
                       os.path.join(path, f"{filename}.fine.json"),
                       sort_keys=sort_keys, mode = mode)

            if self.test_recombination:
                self.test_recomb(caSe, coarse_data, fine_data, filename)
        else: 
            write_json(caSe,
                       os.path.join(path, f"{filename}.json"),
                       sort_keys=sort_keys, mode = mode)


    def addresil(self, new_case):
        if 'tracks' in new_case:
            new_case['tracks']['resources']['resilient'] = "dbg/" 


    def run_experiments(self, force = False):
        # sequentially
        # init the pipeline

        self.TELE.print(f"BEGIN Running {len(self.TODOcases)} cases in experiment {self.experiment_id}.")
        self.i = 0 

        self.job = Job(self.pipeline)

        while bool(self.TODOcases):
            # self.tele.reset_log() # TODO: check why this doesn't work - every new case continues to write the log where previous left off 
            # pop until empty
            newCase = self.TODOcases.pop(0)

            newCase = self.stamp_case(newCase)

            status = self.job.execute_case(newCase)
            
            #dump the TELE of the bench to disk after every case - 
            #otherwise if run fails at some case, the whole bench log is lost
            self.TELE.collect_log(self.EXPERMT)
            self.write_config(self.EXPERMT, "experiment", mode='w+')

            if status == "Fatal":
                self.TELE.error("fatal error in run_case. aborting run")
                # self.TODOcases += newCase
                return
                # TODO: make it graceful

            self.save_case(self.job.CAse) # TODO: decide who should be saving the case

            self.save_output()

            self.DONEcases += [self.job.CAse]

            self.i+=1

        self.TELE.print("Benchmark run complete!")
        self.write_config(self.EXPERMT, "experiment", mode='w+')


    def stamp_case(self, newcase: Case):
        # assigns all the ids and timestamps to the case
        # those depend on time of run of the case

        run_epoch = T.now()

        tstp = T.timestamp(run_epoch)

        case_sign = newcase.ID["case_signature"]

        # TODO: tbh, it should be called run_id or case_run, as the tstp is the time of running of 
        # this case in the benchmark. 
        # the intention of adding tstp to case_id was to make it unique to 
        # avoid unintentionally overwriting the previous runs of the same case.
        # 
        # The case_signature is already supposed to be unique,
        # unless the case is re-run (due to failure in previous run) 
        # then there are possibilites:
        # - the data of failed run is overwritten with new run 
        # - the data of failed run is saved beside the data of new run - with different tstp

        # in the end i decided to leave both. 
        # the user can decide according to their needs
        # which value to use for naming their files

        ID = {
            "experiment_id": self.experiment_id,
            "case_signature": case_sign,
            "case_id": case_sign +"."+ tstp,
            "run_id": case_sign +"."+ tstp,
            "case_index": self.i,
            "output_root": self.output_root,
        }

        summary = {
            "start_time": run_epoch,
            "timestamp": tstp,
        }

        newcase.update_case(".summary", summary)
        newcase.update_case(".ID", ID)

        if self.i == 0:
            self.TELE.print("Starting first case in this run:\n",
                            newcase.case_signature)
        else:
            sep = "=" * 100
            self.TELE.print("\n", sep, "\nStarting next case",
                            newcase.ID['case_index'],
                            ":\n", newcase.case_signature )

        return newcase




    def save_case(self, newCaseExecuted: Case):
        #case_id = case.ID["case_id"]
        #experiment_run = case.results
        # newCaseExecuted = self.tele.results()
        case_id = self.job.CAse.ID["case_id"]

        self.write_config(newCaseExecuted, f"{case_id}.json", self.bench_folder)
        self.TELE.print(f"saving benchmark data to: {case_id}.json")


    def save_output(self):
        # TODO: should also call pipeline's function - 
        # it's the that's supposed to know how to save the output
        # if not defined - fall back to json save
        output = self.job.pipeline.results()
        case_id = self.job.CAse.ID["case_id"]
        self.TELE.print(f"saving output data to: {self.output_folder}/{case_id}.json")
        self.write_config(output, f"{case_id}.json", self.output_folder)


    def save_other(self,v):
        #TODO:temporary hack, shold be intergrated into resilient monitor
        data = v["data"]
        cid = self.CAse.summary["case_id"]
        pth = f"dbg/{cid}"
        os.makedirs(pth, exist_ok=True)

        for prof in data:
            id = prof["id"]
            fnm = prof["function_name"]
            with open(pth + f"/{fnm}_{id}.prof", "w+") as fl:
                fl.write(prof["profile_log"])



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

