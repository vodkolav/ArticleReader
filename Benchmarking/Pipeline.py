from abc import abstractmethod
from Benchmarking.telemetry_manager import TelemetryManager

class Pipeline:

    def __init__(self, output_dir = "output", 
                 checkpoints_dir="checkpoints", 
                 patt = "*"):
        self.delamination_spec = []
        pass
    
    @abstractmethod
    def execute(self, new_case):
        pass
    
    @abstractmethod
    def results(self):
        return self.tele.results()

    @abstractmethod
    def set_telemetry(self, tele: TelemetryManager):
        pass