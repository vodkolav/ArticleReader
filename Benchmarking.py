import time
from datetime import datetime
import threading
import psutil
import os
from ArticleReader.Narrator import Narrator
import torch
from ArticleReader.LatexToSpeech import Chunker
import pandas as pd
from speechbrain.inference import Tacotron2, HIFIGAN


class MemoryMonitor:
    """
    Instance-based memory monitor to track CPU memory usage during inference.
    Each instance keeps its own memory log.
    """
    
    def __init__(self, stage, model_id):
        self.memory_log = []
        self.exception = None
        self.stop_event = threading.Event()
        self.stage = stage
        self.model_id = model_id

    def monitor_cpu_memory(self, interval):
        process = psutil.Process(os.getpid())
        while not self.stop_event.is_set():
            current_memory = process.memory_info().rss
            # here we can add other parameters if need be
            self.memory_log.append({"time": time.time(), "memory": current_memory})
            time.sleep(interval)

    def attach_to(self, forward_func):
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

    def summarize_profile(self, model_profiler):
        data = pd.DataFrame(model_profiler.memory_log)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        if len(data.time)>0:
            dur = (data.time.max() - data.time.min()).total_seconds() 
            #dur = str(dur.microseconds/1e6)
        else:
            dur=0
            
        res = {
            "model_id": model_profiler.model_id ,  #(name)
            "stage": model_profiler.stage,
            "max_memory_use": data['memory'].max(),
            "run_time_sec": dur,
            "memory_log": model_profiler.memory_log,
            "exceptions": model_profiler.exception,
            "n_threads": None
        }
        return res

    def run_experiment(self, processed, case):
        """
        case = {"device": "CPU", 
                "tts_model": "tts-tacotron2-ljspeech",
                "vocoder_model": "tts-hifigan-ljspeech",
                "batch_size": 2, 
                "chunk_length": 50 
       }
        """
        tstp = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
        case_file = "benchmark/" + tstp
        
        self.chunker = Chunker(max_len=case["chunk_length"])
        self.chunker.split_text_into_chunks(processed)
        
        fr = 0 # beginning from chunk
        chunks = self.chunker.get_test_batch(case["batch_size"], fr)        
        self.chunker.save_chunks_as_text(case_file + ".md", chunks)
        
        provider = "speechbrain"

        dev = "cuda" if case["device"]=="GPU" else "cpu"

        model_name = case["tts_model"]
        tts_model = Tacotron2.from_hparams(
                source=f"{provider}/{model_name}",
                savedir=f"checkpoints/{model_name}",
                overrides={"max_decoder_steps": 2000},
                run_opts={"device":dev} 
        )
        tts_model.id = model_name

        model_name = case["vocoder_model"]
        vocoder_model = self.vocoder = HIFIGAN.from_hparams(
                source=f"{provider}/{model_name}",
                savedir=f"checkpoints/{model_name}",
                run_opts={"device":dev} 
            )
        vocoder_model.id = model_name

        self.tts_profiler = MemoryMonitor(stage="tts", model_id=tts_model.id)
        tts_model.encode_batch = self.tts_profiler.attach_to(tts_model.encode_batch)

        self.vocoder_profiler = MemoryMonitor(stage="vocoder", model_id=vocoder_model.id)
        vocoder_model.decode_batch = self.vocoder_profiler.attach_to(vocoder_model.decode_batch)
        

        self.narrator = Narrator(tts_model, vocoder_model) 
        waveforms, durations = self.narrator.text_to_speech_batched(chunks)
        waveform = torch.cat(waveforms, dim=1)
        self.narrator.save_audio(case_file + ".wav", waveform)
        
        sampling_freq = 22050.0
        durations_sec = (durations / sampling_freq).tolist()
        perc_sile = 1- sum(durations)/(max(durations)*len(durations))

        
        parameters = {
            "time": tstp,
            "experiment_id": tstp,
            "chunk_durations": durations_sec,
            "avg_percent_silence": perc_sile
        }
        case.update(parameters)
        
        tts_stage = self.summarize_profile(self.tts_profiler)
        tts_stage.update(case)
        
        voc_stage = self.summarize_profile(self.vocoder_profiler)
        voc_stage.update(case)

        return [tts_stage, voc_stage]



                                   
