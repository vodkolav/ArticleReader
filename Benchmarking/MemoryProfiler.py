import io
import time
from memory_profiler import profile
from typing import Callable, Any, Dict, List, Union
# Define the structure for a single profiling record
# The type hint indicates keys are strings, and values are expected to be 
# int, float, or string (ProfileRecord is a Dict with keys as str and values 
# that are one of the types in the Union).
ProfileRecord = Dict[str, Union[int, float, str]]

class MemoryProfiler:
    """
    A utility class to programmatically wrap a function for memory profiling, 
    collecting results and metadata into a list of unified profile dictionaries.
    
    It tracks a global call ID and a function-specific call count.
    """
    def __init__(self, sampling_type = "interval_sec", sampling_value = 0.1, **kwargs):
        # Stores all collected data records (the main output array)
        self.profiles: List[ProfileRecord] = []
        # Tracks the sequential ID across all attached function calls
        self._call_counter = 0 
        # Tracks the count for each unique function name (e.g., {'func_A': 5})
        self.function_call_counts: Dict[str, int] = {} 
        self.sampling_type = sampling_type
        self.interval = sampling_value

    def attach_to(self, target_func: Callable) -> Callable:
        """
        Wraps the target function with memory-profiler logic and returns the 
        newly wrapped function.

        Args:
            target_func: The function to be profiled.

        Returns:
            The wrapped (instrumented) function.
        """
        profiler = self
        func_name = target_func.__name__  # Capture the function name
        
        def wrapper(*args, **kwargs) -> Any:
            # --- Tracking for this specific call ---
            
            # 1. Update the global ID
            call_id = profiler._call_counter
            profiler._call_counter += 1
            
            # 2. Update the function-specific call count
            profiler.function_call_counts[func_name] = \
                profiler.function_call_counts.get(func_name, 0) + 1
            
            # Retrieve the current function-specific count for this record
            func_call_count = profiler.function_call_counts[func_name]
            
            # 3. Prepare to capture output
            log_stream = io.StringIO()
            
            # 4. Wrap the function with the profiler's logic
            wrapped_func = profile(
                func=target_func, 
                stream=log_stream
            )
            
            # --- Profiling and Timing Start ---
            start_time = time.time()
            
            try:
                # 5. Execute the wrapped function
                result = wrapped_func(*args, **kwargs)
                
            except Exception as e:
                # Record data on failure before re-raising
                end_time = time.time()
                profiler._record_profile(
                    call_id, func_name, func_call_count, start_time, end_time, log_stream.getvalue()
                )
                raise e
            
            # --- Profiling and Timing End ---
            end_time = time.time()
            
            # 6. Record the complete profile data
            profiler._record_profile(
                call_id, func_name, func_call_count, start_time, end_time, log_stream.getvalue()
            )
            
            # 7. Return the original function's result
            return result
        
        return wrapper


    def _record_profile(self, call_id: int, func_name: str, func_call_count: int, start_time: float, end_time: float, log_content: str):
        """Internal method to create and record a single, unified profile record."""
        record: ProfileRecord = {
            "id": call_id,
            "start_time_epoch": start_time,
            "end_time_epoch": end_time,
            "duration_seconds": end_time - start_time,
            "function_name": func_name,
            "function_call_count": func_call_count,
            "profile_log": log_content.strip() # Store the stripped text output
        }
        self.profiles.append(record)


    def clear(self):
        #Clears data storage, but attached functions are still connected 
        self.profiles: List[ProfileRecord] = []
        # Tracks the sequential ID across all attached function calls
        self._call_counter = 0 
        # Tracks the count for each unique function name (e.g., {'func_A': 5})
        self.function_call_counts: Dict[str, int] = {} 


    def summarize(self) -> List[ProfileRecord]:
        """
        Returns the array of all collected profiling data records.
        """

        res = {
            "sampling_type": self.sampling_type,
            "sampling_value": self.interval,
            "data": self.profiles,
            "summary": {
                "n_samples": len(self.profiles),
                    }
            }
        
        self.clear()
        return res

