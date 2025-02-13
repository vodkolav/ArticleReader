import time
from datetime import datetime
import threading
import psutil
import os
import torch
from ArticleReader.LatexToSpeech import Narrator, Chunker
import pandas as pd


class MemoryMonitor:
    """
    Instance-based memory monitor to track CPU memory usage during inference.
    Each instance keeps its own memory log.
    """
    
    def __init__(self, stage):
        self.memory_log = []
        self.exception = None
        self.stop_event = threading.Event()
        self.stage = stage

    def monitor_cpu_memory(self, interval):
        process = psutil.Process(os.getpid())
        while not self.stop_event.is_set():
            current_memory = process.memory_info().rss
            self.memory_log.append({"time": time.time(), "memory": current_memory})
            time.sleep(interval)

    def monitor_memory_decorator(self, forward_func):
        def wrapper(model, *args, **kwargs):
            self.memory_log.clear()  # Clear previous logs
            interval = 0.1
            
            # Start the CPU memory monitoring thread.
            monitor_thread = threading.Thread(target=self.monitor_cpu_memory, args=(interval,))
            monitor_thread.start()
            try:
                output = forward_func(model, *args, **kwargs)  # Run the original forward pass
            except Exception as e:
                self.exception = e.message
                output = None
            # Stop monitoring
            finally:
                self.stop_event.set()
                monitor_thread.join()
            
            return output
        return wrapper


class Bench:

    def per_model(self, model_profiler):
        data = pd.DataFrame(model_profiler.memory_log)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        if len(data.time)>0:
            dur = (data.time.max() - data.time.min()).total_seconds() 
            #dur = str(dur.microseconds/1e6)
        else:
            dur=0
            
        res = {
            "model_id": "taco",  #(name)
            "stage": model_profiler.stage,
            "max_memory_use": data['memory'].max(),
            "run_time_sec": dur,
            "memory_log": model_profiler.memory_log,
            "exceptions": model_profiler.exception,
            "n_threads": None
        }
        return res

    def run_experiment(self, processed, case):
        tstp = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
        case_file = "benchmark/" + tstp
        
        chunker = Chunker(max_len=case["chunk_length"])
        chunker.split_text_into_chunks(processed)
        
        fr = 0 # beginning from chunk
        chunks = chunker.get_test_batch(case["batch_size"], fr)
        
        chunker.save_chunks_as_text(case_file + ".md", chunks)
        
        narrator = Narrator(profiling=1) # TODO: define model 
        waveforms, durations = narrator.text_to_speech_batched(chunks)
        durations_sec = (durations / 22050.0).tolist()
        perc_sile = 1- sum(durations)/(max(durations)*len(durations))
        waveform = torch.cat(waveforms, dim=1)
        narrator.save_audio(case_file + ".wav", waveform)
        
        parameters = {
            "time": tstp,
            "experiment_id": tstp,
            "chunk_durations": durations_sec,
            "avg_percent_silence": perc_sile
        }
        case.update(parameters)
        
        tts_stage = self.per_model(narrator.profilers['tacotron'])
        tts_stage.update(case)
        
        voc_stage = self.per_model(narrator.profilers['vocoder'])
        voc_stage.update(case)

        return [tts_stage, voc_stage]



                                   
