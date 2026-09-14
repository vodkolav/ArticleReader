from Benchmarking.Case import Case
from Benchmarking.telemetry_manager import TelemetryManager
from Benchmarking.timeutils import Time as T

from Benchmarking.Pipeline import Pipeline

import os
from copy import deepcopy
import logging

class Job: # or worker? for multithreading...

    tele: TelemetryManager # telemetry for current case

    pipeline: Pipeline

    CAse: Case

    bench = None


    @property
    def onerror(self):
        return self.bench.onerror


    def __init__(self, bench):

        self.CAse = Case({})
        self.tele = TelemetryManager(self.CAse)
        self.tele.intercept_logging(log_level=logging.WARNING)
        self.bench = bench
        self.pipeline = bench.pipeline()
        self.pipeline.set_telemetry(self.tele)


    #TODO: think about whole lifecycle of the log (and other tracked data)
    # when it's loaded from previous runs and continued. 
    # including re-run of individual cases.


    def execute_case(self, new_case):
        try:
            
            self.init_case(new_case)  # deals with configs
            self.pipeline.init_telemetry() # deals with tracks
            self.pipeline.run_case()
            status = "Ok"

        except Exception as e:

            cid = self.CAse.ID["case_id"]
            self.tele.error("Error executing case", cid, ":", str(e), e)
            status = "Error"
            #TODO: Set the case.summary.status to 'error', so that it can be queried in the final report data

            if self.onerror == "fail":
                raise

        finally:
            cid = self.CAse.ID["case_id"]
            #TODO: these might fail as well. handle that gracefully
            self.close_case()
            try:
                self.pipeline.post_case()
            except Exception as ee:
                self.tele.error("Error producing post-processing data for case:", cid, ee)
        return status


    def init_telemetry(self, obj, new_case: Case):
        # re-runs for every new case
        # TODO: attach monitors for particular pipeline components 

        if obj == None:
            return # no object to attach to 
        
        self.tele.CAse = new_case
        tracks = new_case['tracks']

        for path, conf in self.tracks_configs(tracks):

            self.tele.AttachSensor(obj, path=".tracks." + path, **conf)
            obj.tele = self.tele 


    def tracks_configs(self, tracks):
        #TODO: this method is pretty flimsy. 
        # it relies on existence of 'func_name' key in the 
        # tracks data, which is not always the case.
        # need to find a more concrete anchor for identifying the sensor path.
        for snsr, snsconf in tracks.items():
            if "func_name" in snsconf.keys():
                yield snsr, snsconf
            else:
                for k, v in snsconf.items():
                    k = snsr + "." + k
                    yield k , v


    def init_case(self, new_case: Case):

        force = False
        if not self.CAse:
            self.CAse = new_case
            force = True  # first run, so all initializers must run

        elif self.CAse.config == new_case.config:
            self.tele.warning("all configs are already identical, which should not happen")  # raise Error?;  


        if 'data' in new_case.get_path('tracks.harvest.curves').keys():
            #TODO: make this check not dependent of specific path above
            self.tele.warning("There is already data in the new case. This should not happen")

        #Check for all initializers, whether their value changed and 
        #re-init whichever have and all their downstream initializers

        for key, init_func in self.pipeline.initializers.items():
            try:
                cur_val = self.CAse.get_path(key)
                new_val = new_case.get_path(key)
            except KeyError as e:
                raise ValueError(f"Case is missing required key: {key}")

            different = cur_val != new_val
            if different or force:
                force = True # once a change is detected, all downstream initializers must run
                if not different:
                    self.tele.print(f" initializing {key} to {new_val}")
                else:
                    self.tele.print(f" re-initializing {key} from {cur_val} to {new_val}")
                # self.CAse.update_case(key, new_val)
                obj = init_func(new_case)
                self.init_telemetry(obj, new_case)
            else:
                continue  # already initialized to the same value
        force = False

        self.CAse = new_case


    def close_case(self):

        #result.update(models_result)
        self.CAse.summary["end_time"] = T.now()
        self.tele.collect_sensors(self.CAse)
        case_indx = self.CAse.ID["case_index"]
        case_sign = self.CAse.ID["case_signature"]
        self.tele.print("Case", case_indx, "Done:\n", case_sign )
        self.tele.collect_log(self.CAse)
        self.tele.reset_log()
        print('hi')


    # def results(self):
    #     return deepcopy(self.CAse.results)