from Benchmarking.telemetry_manager import TelemetryManager

class Pipeline:

    def __init__(self, output_dir = "output", 
                 checkpoints_dir="checkpoints", 
                 patt = "*"):
        pass
    
    def run_case(self, new_case):
        pass

    def set_telemetry(self, telem: TelemetryManager):
        self.tele = telem
        self.run_case = self.tele.add_memory_monitor(self.run_case, "" )
