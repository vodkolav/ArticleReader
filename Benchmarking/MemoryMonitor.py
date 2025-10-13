import pandas as pd
import psutil


import os
import resource
import threading
import time


class MemoryMonitor:
    """
    Instance-based memory monitor to track CPU memory usage during inference.
    Each instance keeps its own memory log and dynamically adjusts memory limits if needed.
    """

    def __init__(self, interval_sec: float = 0.1, **kwargs): #, stage, model_id):
        self.memory_log = []
        self.exception = None
        self.stop_event = threading.Event()
        # memory monitor needs no metadata on monitored object
        # self.stage = stage
        # self.model_id = model_id
        self.process = psutil.Process(os.getpid())
        self.memory_limit_bytes = self.get_free_memory_bytes()*1.2 #20000 # 40000
        self.last_process_count = 0
        self.interval = interval_sec
        self.init_rss_mb = self.get_memory_usage_mb()


    def get_free_memory_bytes(self):
        with open('/proc/meminfo', 'r') as mem:
            free_memory = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    free_memory += int(sline[1])
        return free_memory * 1000

    def get_process_group(self):
        pgid = os.getpgid(os.getpid())
        group = []
        for p in psutil.process_iter():
            try:
                if os.getpgid(p.pid) == pgid:
                    group.append(p)
            except Exception:
                print("do we really want to pass? 1")
                pass
        return group
    
    def get_memory_usage_mb(self):
        """
        Returns the current process's resident set size (RSS) memory usage in MB using psutil.
        Works cross-platform.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # RSS in megabytes

    def siblings_snapshot(self):
        # heavy artillery, not tested much. use sparingly.
        # snapshots the stats of all the sibling processes of this process
        siblings = self.process.parent().children(recursive=True)
        snpsh = []
        for s in siblings:
            snp = s.memory_info()._asdict()
            snp["_pid"] = s._pid
            snp["_name"] = s._name
            snpsh.append(snp)
        return snpsh

    def set_memory_limit(self, all_processes):
        """Estimate process count and apply memory limits only if needed."""

        num_processes = max(1, len(all_processes))
        per_process_limit = (self.memory_limit_bytes // num_processes)

        try:
            # Check process count every 5 seconds and adjust if needed
            if len(self.memory_log) % int(5 / self.interval) == 0:

                # Update limits only if the number of processes has changed significantly
                if abs(num_processes - self.last_process_count) / max(1, self.last_process_count) > 0.2:
                    self.last_process_count = num_processes
                    for p in all_processes:
                        try:
                            p.rlimit(resource.RLIMIT_AS, (per_process_limit, resource.RLIM_INFINITY))
                            #resource.setrlimit(resource.RLIMIT_AS, (per_process_limit, resource.RLIM_INFINITY))
                        except Exception:
                            print("do we really want to Ignore permission errors?")
                            pass  # Ignore permission errors
        except Exception:
            print("do we really want to Ignore rare process termination errors?")
            pass  # Ignore rare process termination errors
        return num_processes, per_process_limit

    def monitor_cpu_memory(self):
        while not self.stop_event.is_set():
            all_processes = self.get_process_group()
            #all_processes = self.process.children(recursive=True) + [self.process]

            num_processes, per_process_limit = self.set_memory_limit(all_processes)

            RSS = sum(p.memory_info().rss for p in all_processes)
            VMS = sum(p.memory_info().vms for p in all_processes)
            # here we can add other parameters if need be
            self.memory_log.append({"time": time.time(),
                                    #"memory": RSS,
                                    "RSS_mb": RSS,
                                    "VMS_mb": VMS,
                                    "processes": num_processes,
                                    "num_threads": self.process.num_threads(),
                                    "per_process_limit":per_process_limit,
                                    "free_memory":self.get_free_memory_bytes(),
                                    #"siblings": self.siblings_snapshot() #only use when REALLY needed
                                    })


            time.sleep(self.interval)

    def attach_to(self, forward_func):
        def wrapper(model, *args, **kwargs):
            self.memory_log.clear()  # Clear previous logs
            # add first empty record to split the 
            # different cases on the graphs   
            self.memory_log.append({"time": time.time()}) #           
            # Set initial memory limit
            #self.set_memory_limit()

            # Start the CPU memory monitoring thread.
            monitor_thread = threading.Thread(target=self.monitor_cpu_memory)
            monitor_thread.start()

            try:
                output = forward_func(model, *args, **kwargs)  # Run the original forward pass
            except Exception as e:
                print("forward_func failed with exc: ", str(e))
                self.exception = str(e)
                output = None
            # Stop monitoring
            finally:
                # Ensure monitoring stops even if an exception occurs
                self.stop_event.set()
                monitor_thread.join()

            return output
        return wrapper


    def summarize(self):

        if len(self.memory_log)>1:
            # data = pd.DataFrame(self.memory_log)
            # data['time'] = pd.to_datetime(data['time'], unit='s')
            # dur = (data.time.max() - data.time.min()).total_seconds()
            a = self.memory_log[0]['time']
            b = self.memory_log[-1]['time']
            dur = b-a
            memuse = 100500 # TODO: track this during monitoring 
            #  data['memory'].max()
            #dur = str(dur.microseconds/1e6)
        else:
            dur=0
            memuse=None

        res = {
            # "model_id": self.model_id ,  #(name)
            # "stage": self.stage,
            "sampling_type": "interval_sec",
            "sampling_value": self.interval,
            "data": self.memory_log,
            "summary": {
                "n_samples": len(self.memory_log),
                "duration_sec": dur,
                "max_memory_use_mb": memuse,
                "exceptions": self.exception,
                "init_rss_mb": self.init_rss_mb,
                "memory_limit_bytes": self.memory_limit_bytes,
                "n_threads": None
                    }
            }
        return res