from abc import abstractmethod
from Benchmarking.telemetry_manager import TelemetryManager

class Pipeline:

    tele: TelemetryManager
    initializers: dict

    def __init__(self, output_dir = "output", 
                 checkpoints_dir="checkpoints", 
                 patt = "*"):
        self.delamination_spec = []
        pass
    
    @abstractmethod
    def run_case(self, new_case):
        pass
    
    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def set_telemetry(self, tele: TelemetryManager):
        pass

    @abstractmethod
    def init_telemetry(self):
        pass