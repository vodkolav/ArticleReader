

from Benchmarking import utils as butils
from Benchmarking.Case import Case
from Benchmarking.timeutils import Time as T

import os
import numpy as np
import logging
from copy import deepcopy


class DummyTelemetryManager:
    # TODO: maybe make it an abstract base class, and have TelemetryManager inherit from it.
    def __init__(self):
        pass

    def print(self, entry: dict):
        pass

    def error(self):
        pass

    def ping(self):
        pass

    def warning(self):
        pass


class TelemetryManager:
    #TODO: rename to just Telemetry
    #TODO: implement proper logging
    #TODO: implement progression percentage of individual cases and the whole experiment
    """
    Manages the collection of training and evaluation metrics.
    """

    log: list
    sensors: dict = {}
    _CAse: Case

    @property
    def CAse(self):
        return self._CAse

    @CAse.setter
    def CAse(self, value):
        # if not isinstance(value, Case):
        #     raise ValueError("CAse must be an instance of Case class.")
        self._CAse = value

        #TODO: kinda ugly, should use json paths
        # also should think about whole lifecycle of the log (and other tracked data)
        # when it's loaded from previous runs and continued. 
        # including re-run of individual cases.

        log = self.CAse.tracks.get("log",None)
        if log: 
            self.log = log.get("data", []) 


    def __init__(self, log_level=logging.WARNING): # experiment_config: dict ):

        # sensors are independent components that 
        # usually track system resources, such as memory/GPU etc.
        # run on separate threads
        # not in sync with telemetry episodes/epochs

        self._CAse: Case = []

        self.log = []

        self.lastType = "info"

        #TODO: define float format, ex: Avg Reward (last 100): {avg_reward:.2f}

        #self.tstp_format = "%Y%m%d-%H%M%S-%f"
        self.tstp_format = "%Y%m%d-%H%M%S"


    def intercept_logging(self, log_level=logging.WARNING):
        """enables interception of all system logs by this telemetry.

        Args:
            log_level (_type_, optional): logging level to intercept. Defaults to logging.WARNING.
        """

        self.print("TelemetryManager: intercepting system logs at level:", logging.getLevelName(log_level))

        NPERRSTATE = 'warn'
        # This acts as the global safety net across your entire app 
        np.seterr(divide=NPERRSTATE, invalid=NPERRSTATE, over=NPERRSTATE, under=NPERRSTATE)
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

        # REMOVE CONSOLE DOUBLE-PRINTING:
        # Wipe out any default handlers (like StreamHandler) that Python auto-creates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        root_logger.addHandler(telemetry_handler)


    def reset_log(self):
        self.log = []


    def print(self, *what):
        #TODO: rename to info
        self._print(*what, type = "info")


    def Log(self, entry:dict): 
        self.log.append(entry)


    def _print(self, *what, type = 'info', **kwargs):

        #TODO: make this make sense
        what = " ".join([str(w) for w in  what])

        what = what + kwargs['message'] if 'message' in kwargs else what

        time = kwargs["time"] if 'time' in kwargs else T.now()
        entry = {
            "type": type,
            "time": time,
            "message": what,
            }
        
        if 'source' in kwargs:
            entry['source'] = kwargs['source']

        self.Log(entry)

        nl = kwargs["newline"] if 'newline' in kwargs else True

        message = self.format_entry(entry)

        if self.lastType == "ping" and type != "ping":
            message ="\n" + message
            # self.display("", newline=True)

        self.display(message, newline=nl)
        self.lastType = type


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
        return disp.format(**newentry).replace("\n", " ")


    def display(self, what, newline = False) -> None:
        if newline:
            print(what)
        else:
            print(f"\r{what}", " "*100 , end='')


    def ping(self, *what) -> None:
        if self.lastType == 'ping':
            self._print(*what, type= 'ping', newline=False)
        else:
            self._print(*what, type= 'ping', newline=True)
        self.lastType = 'ping'


    def collect_log(self):
        self.CAse.update_case(".tracks.log", {})
        self.CAse.update_case(".tracks.log.data", self.log)
        # TODO: make log one of the sensors? 


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
                # summ_func = getattr(obj, kwargs['summary_func'])
                func = snsr.attach_to(func)

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


    def results(self):
        return deepcopy(self.CAse.results)



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
            "message" : message,
            "type" : record.levelname.lower(),
            "time" : record.created,
            "source": record.name # 'speechbrain.utils.fetching'
        }
        # other properties from record can be added

        # Pass the formatted message and/or the raw record to your manager
        # The exact method call depends on your manager's API
        self.manager._print(**entry)

