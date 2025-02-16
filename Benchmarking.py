import time
from datetime import datetime
import threading
import psutil
import os
from ArticleReader.Narrator import Narrator
import torch
from ArticleReader.Chunker import Chunker
import pandas as pd
from speechbrain.inference import Tacotron2, HIFIGAN
import json

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

        if len(model_profiler.memory_log)>1:
            data = pd.DataFrame(model_profiler.memory_log)
            data['time'] = pd.to_datetime(data['time'], unit='s')
            dur = (data.time.max() - data.time.min()).total_seconds() 
            memuse = data['memory'].max()
            #dur = str(dur.microseconds/1e6)
        else:
            dur=0
            memuse=None
            
        res = {
            "model_id": model_profiler.model_id ,  #(name)
            "stage": model_profiler.stage,
            "max_memory_use": memuse,
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

        
        result = {
            "time": tstp,
            "experiment_id": tstp,
            "chunk_durations": durations_sec,
            "avg_percent_silence": perc_sile
        }
        result.update(case)
        
        tts_stage = self.summarize_profile(self.tts_profiler)
        tts_stage.update(result)
        
        voc_stage = self.summarize_profile(self.vocoder_profiler)
        voc_stage.update(result)

        return [tts_stage, voc_stage]



    def run_experiments(self, processed_text, grid):
        """
        grid = {"chunk_length": range(50, 500, 50),
                "batch_size": (1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 200),
                "tts_model": ["tts-tacotron2-ljspeech"],
                "vocoder_model": ["tts-hifigan-ljspeech"],
                "device": ["CPU"], 
            }
        grid
        """        
        self.provider = "speechbrain"
        
        self.case_objects = {}
        self.case = {}

        for d in grid["device"]:
            self.init_device(d)

            for tts_model_name in grid["tts_model"]:                
                self.init_tts_model(tts_model_name)

                for voc_model_name in grid["vocoder_model"]:     
                    self.init_voc_model(voc_model_name)

                    for chunk_length in grid["chunk_length"]:
                        self.init_chunker(processed_text, chunk_length)

                        for batch_size in grid["batch_size"]:
                            self.init_batch(batch_size)


                            print(f"running experiment:\n {json.dumps(self.case)}")
                            wat = json.dumps(self.case_objects["batch_size"])
                            print(f"test data:\n {wat}")
                            experiment_run = self.run_case()
                            with open("benchmark/" + experiment_run[0]["experiment_id"] + ".json", "w+") as f:
                                json.dump(experiment_run,f)

    def init_batch(self, batch_size):
        #TODO: take batches of sorted chunks
        fr = 25 # beginning from chunk
        batch = self.chunker.get_test_batch(batch_size, fr)        
                          
        self.case_objects["batch_size"] = batch
        self.case["batch_size"] = batch_size

    def init_chunker(self, processed_text, chunk_length):
        self.chunker = Chunker(max_len=chunk_length)
        self.chunker.split_text_into_chunks(processed_text)                        
        self.case_objects["chunk_length"] = self.chunker
        self.case["chunk_length"] = chunk_length

    def init_voc_model(self, voc_model_name):
        vocoder_model = self.vocoder = HIFIGAN.from_hparams(
                        source=f"{self.provider}/{voc_model_name}",
                        savedir=f"checkpoints/{voc_model_name}",
                        run_opts={"device":self.case_objects["device"]} 
                        )
        vocoder_model.id = voc_model_name

        self.vocoder_profiler = MemoryMonitor(stage="vocoder", model_id=vocoder_model.id)
        vocoder_model.decode_batch = self.vocoder_profiler.attach_to(vocoder_model.decode_batch)
                    
        self.case_objects["vocoder_model"] = vocoder_model
        self.case["vocoder_model"] = voc_model_name

    def init_device(self, d):
        dev = "cuda" if d =="GPU" else "cpu"            
        self.case_objects["device"] = dev
        self.case["device"] = d
        return dev

    def init_tts_model(self, tts_model_name):

        tts_model = Tacotron2.from_hparams(
                    source=f"{self.provider}/{tts_model_name}",
                    savedir=f"checkpoints/{tts_model_name}",
                    overrides={"max_decoder_steps": 2000},
                    run_opts={"device":self.case_objects["device"]} 
                    )
        tts_model.id = tts_model_name
        self.tts_profiler = MemoryMonitor(stage="tts", model_id=tts_model.id)
        tts_model.encode_batch = self.tts_profiler.attach_to(tts_model.encode_batch)

        self.case_objects["tts_model"] = tts_model
        self.case["tts_model"] = tts_model_name

    def run_case(self):

        sampling_freq = 22050.0
        tstp = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
        case_file = "output/" + tstp

        device = self.case_objects["device"]
        tts_model = self.case_objects["tts_model"]
        vocoder_model = self.case_objects["vocoder_model"]
        chunk_length = self.case_objects["chunk_length"]
        batch= self.case_objects["batch_size"]


        self.narrator = Narrator(tts_model, vocoder_model) 
        waveforms, durations = self.narrator.text_to_speech_batched(batch)
        waveform = torch.cat(waveforms, dim=1)
        self.narrator.save_audio(case_file + ".wav", waveform)

        self.chunker.save_chunks_as_text(case_file + ".md", batch)

        # create a report
        durations_sec = (durations / sampling_freq).tolist()
        perc_sile = 1- sum(durations)/(max(durations)*len(durations))
        
        result = {
            "time": tstp,
            "experiment_id": tstp,
            "chunk_durations": durations_sec,
            "avg_percent_silence": perc_sile
        }
        result.update(self.case)
        
        tts_stage = self.summarize_profile(self.tts_profiler)
        tts_stage.update(result)
        
        voc_stage = self.summarize_profile(self.vocoder_profiler)
        voc_stage.update(result)

        return [tts_stage, voc_stage]


    def test_permutations(self):

        grid = {"A": [1,2,3,4,5,6],
             "B": "a b c d e f g h i j".split(' '),
             "C": ["U", "V"],
             "D": ["J","K"],
             "E": ["P"], 
               }
        res = self.permutations(grid)
        
        import json
        with open("check.json", 'w+') as f: 
            json.dump(res, f, indent=4)


        import pandas as pd 
        df = pd.read_json("check.json")
        print(df)

        print(len(df.drop_duplicates())) 

    def permutations(self, grid):

        keys = list(grid.keys())
        n = len(keys)

        layers = [[{keys[l]:v} for v in grid[keys[l]]]  for l in range(n)]

        def combine(prev, this):
            print(prev)
            print(this)
            tmp = [ t.copy() for t in this]
            [th.update(prev) for th in tmp]
            return tmp

        res = layers[0]
        for i in range(1,n):
            l1 = res
            l2 = layers[i]
            res = [combine(l, l2) for l in l1]
            res = sum(res,[])
        return res



                                   
