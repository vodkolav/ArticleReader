
# --- Example Usage ---
import time
import json
from Benchmarking.MemoryProfiler import MemoryProfiler
from Benchmarking.utils import *

# Dummy function to simulate a memory leak or heavy usage
def data_processor(data_list: list, buffer_size_mb: int):
    """Function 1: Allocates and holds memory."""
    # Allocate a buffer to simulate memory usage
    buffer_bytes = buffer_size_mb * 1024 * 1024
    buffer = bytearray(buffer_bytes)
    
    # Simulate a subtle "leak" by holding onto a small piece of data
    global LEAKED_DATA
    LEAKED_DATA = buffer
    
    time.sleep(0.05)
    
    return len(buffer)

def reporting_task(report_id: int):
    """Function 2: A different, smaller task."""
    time.sleep(0.01)
    _ = [0] * (report_id * 1000)
    return f"Report {report_id} done"

LEAKED_DATA = None 

if __name__ == '__main__':
    # 1. Instantiate the profiler (a single instance for all functions)
    profiler = MemoryProfiler()

    # 2. Attach the profiler to multiple target functions
    data_processor_wrapped = profiler.attach_to(data_processor)
    reporting_task_wrapped = profiler.attach_to(reporting_task)

    # 3. Call the wrapped functions in an interleaved run
    print("--- Starting Interleaved Profiling Runs ---")
    
    # Run 1: data_processor (Global ID 0, Func Count 1)
    data_processor_wrapped(data_list=["A"], buffer_size_mb=20)
    print("Called data_processor (20MB)")
    
    # Run 2: reporting_task (Global ID 1, Func Count 1)
    reporting_task_wrapped(report_id=1)
    print("Called reporting_task (ID 1)")

    # Run 3: data_processor (Global ID 2, Func Count 2)
    data_processor_wrapped(data_list=["B"], buffer_size_mb=45)
    print("Called data_processor (45MB)")

    # 4. Summarize and inspect the collected data
    summary = profiler.summarize()
    
    print("\n--- Summary of All Profiling Runs ---")
    print(f"Total calls tracked: {len(summary)}") 
    

    write_json(summary, "tests/data/memProf")

    # Use json for clean output of the collected array
    # print(json.dumps(summary, indent=2))