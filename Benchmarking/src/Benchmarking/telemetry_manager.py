

from Benchmarking import utils as butils
from Benchmarking.Case import Case
from Benchmarking.timeutils import Time as T

import os

import logging
from copy import deepcopy


class TelemetryManager:
    #TODO: rename to just Telemetry
    #TODO: implement proper logging
    """
    Manages the collection of training and evaluation metrics.
    """

    log: list
    sensors: dict = {}
    CAse: Case

    def __init__(self, log_level=logging.WARNING): # experiment_config: dict ):
        
        #self.metrics_data = experiment_config

        # sensors are independent components that 
        # usually track system resources, such as memory/GPU etc.
        # run on separate threads
        # not in sync with telemetry episodes/epochs
        # self.sensors = {}

        self.CAse: Case = []

        self.log = []

        #self.summary = {}

        self.lastType = "info"


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

        
        # 2. Redirect standard warnings into the logging framework
        logging.captureWarnings(True)

        # 3. Create the bridge handler (passing 'self' as the telemetry instance)
        # We can define the handler class right below or keep it separate
        telemetry_handler = TelemetryManagerHandler(self)

        # 4. Set formatting
        formatter = logging.Formatter('%(filename)s:%(lineno)d - %(message)s')
        telemetry_handler.setFormatter(formatter)

        # This ensures the log message passed to your manager is formatted correctly
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 5. Attach the handler to the Root Logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)  # Configurable log level
        root_logger.addHandler(telemetry_handler)


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




    def print(self, *what):
        #TODO: rename to info
        self._print(*what, type = "info")


    def Log(self, entry:dict): 
        self.log.append(entry)


    def _print(self, *what, type = 'info', **kwargs):
        if self.lastType == "ping" and type != "ping":
            self.display("", newline=True)
        self.lastType = type
        #TODO: change this
        what = " ".join([str(w) for w in  what])

        time = kwargs["time"] if 'time' in kwargs else T.now()
        entry = {
            "type": type,
            "time": time,
            "message": what,
            }
        self.Log(entry)

        nl = kwargs["newline"] if 'newline' in kwargs else True
        self.display(self.format_entry(entry), newline=nl)


    def warning(self, *what):
        self._print(*what, type = "warning")


    def error(self, *what):
        self._print(*what, type = "error")


    def debug(self, *what):
        self._print(*what, type = "debug")


    def format_entry(self, entry: dict):
        newentry = entry.copy()
        newentry['timestamp'] = T.timestamp(newentry['time'])
        disp = "[{type}] {timestamp}: {message}"
        return disp.format(**newentry)


    def display(self, what, newline = False) -> None:
        if newline:
            print(what)
        else:
            print(f"\r{what}", " "*100 , end='')


    def ping(self, *what) -> None:
        if self.lastType == 'ping':
            self._print(*what, type= 'ping', newline=False)
        else:
            self.lastType = 'ping'
            self._print(*what, type= 'ping', newline=True)


    def start(self, new_case):
        # Start telemetry reporting for an experiment
        #self.summary = self.case.get("summary", {})
        # self.case = new_case
        self.log = []


    # def collect_episodes(self):
    #     self.case["tracks"]["episodes"]["data"] = self.episodes


    def collect_log(self):
        self.CAse.update_case(".tracks.log.data", self.log)
        # TODO: make log one of the sensors? 


    def add_memory_monitor(self, func, config: dict, label):
        monitor = MemoryMonitor(**config)
        func = monitor.attach_to(func)
        self.sensors[label] = monitor
        return func


    def AttachSensor(self, obj, func_name, path, **kwargs):

        level, tracks, sensor, label = butils.nunpack(path.split("."),4)

        if tracks != "tracks":
            raise ValueError("path not recognized: " + path)
        func = getattr(obj, func_name)

        match sensor:
            case "harvest": 
                config = self.CAse.get_path(path)
                from Benchmarking.sensors.Harvester import Harvester
                snsr = Harvester(**config)
                summ_func = getattr(obj, kwargs['summary_func'])
                func = snsr.attach_to(func, summ_func)

            case "episodes": 
                config = self.CAse.get_path(path)
                from Benchmarking.sensors.EpisodeTracker import EpisodeTracker
                snsr = EpisodeTracker(**config)
                summ_func = getattr(obj, kwargs['summary_func'])
                func = snsr.attach_to(func, summ_func)

            case "resources": 
                
                if "config" in kwargs:
                    config = kwargs["config"]
                else:
                    config = self.CAse.get_path(path)
                from Benchmarking.sensors.MemoryMonitor import MemoryMonitor
                snsr = MemoryMonitor(**config)
                #summ_func = kwargs['summary_func']
                func = snsr.attach_to(func)

            case "profile": 
                if  path in self.sensors:
                    snsr = self.sensors[path]
                else:                        
                    if "config" in kwargs:
                        config = kwargs["config"]
                    else:
                        config = self.CAse.get_path(path)
                        # config = butils.get_path(path, self.case, default= {})  
                        # TODO: verify we really don't need 'default' parameter
                    from Benchmarking.sensors.MemoryProfiler import MemoryProfiler
                    snsr = MemoryProfiler(**config)
                func = snsr.attach_to(func)

        setattr(obj,func_name, func)
        snsr.tele = self
        self.sensors[path] = snsr        

            #case "log":  TODO: decide if me make it a sensor too. or leave it special

    def add_EpisodeTracker(self, ep_func, summ_func, config: dict, label):
        trckr = EpisodeTracker(**config)
        ep_func = trckr.attach_to(ep_func, summ_func)
        self.sensors[label] = trckr
        return ep_func


    def collect_sensors(self):

        for k,v in self.sensors.items():            
            sens_summary = deepcopy(v.summarize())            
            if "profile" in k and sens_summary['data']!=[]:
                self.save_other(sens_summary)

            self.CAse.update_case(k, sens_summary)


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


    def end(self):
        # Optional: Print end message
        self.collect_sensors()
        self.collect_log()

        # create a report

        case_indx = self.CAse.ID["case_index"]
        case_sign = self.CAse.ID["case_signature"]
        self.CAse.summary["end_time"] = T.now()
        self.print("Case", case_indx, "Done:\n", case_sign )


    def results(self):
        return deepcopy(self.CAse)


    def progress(self, message = ""):
        # TODO implement progression percentage of individual cases
        # Optional: Print progress
        if self.i_episode in self.samplePoints:            
            msg = f"\r {self.mode} Episode {self.i_episode}/{self.total_episodes}" + message
            self.display(msg)


class TelemetryManagerHandler(logging.Handler):
    """
    A custom handler that sends log records to a specified manager object.
    """
    def __init__(self, manager_object: TelemetryManager):
        super().__init__()
        self.manager = manager_object

    def emit(self, record: logging.LogRecord):
        """
        Called by the logging system for each log record.
        """

        # Format the record before passing it to the manager
        message =  self.format(record)
        # dont remove - it's the line that creates record.message
        # TODO: make it proper
        
        entry = {
            "message" : record.message,
            "type" : record.levelname.lower(),
            "time" : record.created,
            "source": record.name # 'speechbrain.utils.fetching'
        }
        # other properties from record can be added

        # Pass the formatted message and/or the raw record to your manager
        # The exact method call depends on your manager's API
        self.manager.Log(entry)

