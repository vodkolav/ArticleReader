from ArticleReader.Narrator import Narrator
import torch
from ArticleReader.Chunker import Chunker
from speechbrain.inference import Tacotron2, HIFIGAN


class Pipeline:
    def __init__(self):
        pass

class TTSPipeline(Pipeline):
    def __init__(self, benchmark_dir = "benchmark", patt = "*"):
        self.benchmark_dir = benchmark_dir
        self.donecases = self.load_benchmarks(patt) 


    def init_batch(self, batch_size):
        # take batches of sorted chunks
        fr = 0 # beginning from chunk              
        batch = self.chunker.get_batch_sorted(batch_size, fr)    

        #batch = self.chunker.get_test_batch(batch_size, fr)        
        self.case_objects["batch_size"] = batch
        self.case["batch_size"] = batch_size

    def init_chunker(self, processed_text, chunk_length):
        self.chunker = Chunker(max_len=chunk_length)
        self.chunker.split_text_into_chunks(processed_text)                        
        self.case_objects["chunk_length"] = self.chunker
        self.case["chunk_length"] = chunk_length

    def init_voc_model(self, voc_model_name):
        vocoder_model = HIFIGAN.from_hparams(
                        source=f"{self.provider}/{voc_model_name}",
                        savedir=f"checkpoints/{voc_model_name}",
                        run_opts={"device":self.case_objects["device"]} 
                        )
        vocoder_model.id = voc_model_name

        # self.vocoder_profiler = MemoryMonitor(stage="vocoder", model_id=vocoder_model.id)
        # vocoder_model.decode_batch = self.vocoder_profiler.attach_to(vocoder_model.decode_batch)
                    
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
        # self.tts_profiler = MemoryMonitor(stage="tts", model_id=tts_model.id)
        # tts_model.encode_batch = self.tts_profiler.attach_to(tts_model.encode_batch)

        self.case_objects["tts_model"] = tts_model
        self.case["tts_model"] = tts_model_name

    
    def run_case(self):

        sampling_freq = 22050.0
        tstp = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
        case_file = "output/" + tstp

        print(f"{tstp}: running experiment:\n {json.dumps(self.case, indent=2)}")

        device = self.case_objects["device"]
        tts_model = self.case_objects["tts_model"]
        vocoder_model = self.case_objects["vocoder_model"]
        chunk_length = self.case_objects["chunk_length"]
        batch= self.case_objects["batch_size"]

        self.tts_profiler = MemoryMonitor(stage="tts", model_id=tts_model.id)
        tts_model.encode_batch = self.tts_profiler.attach_to(tts_model.encode_batch)

        self.vocoder_profiler = MemoryMonitor(stage="vocoder", model_id=vocoder_model.id)
        vocoder_model.decode_batch = self.vocoder_profiler.attach_to(vocoder_model.decode_batch)

        # TTS
        self.narrator = Narrator(tts_model, vocoder_model)        
        print(" Running text_to_speech_df") 
        batch_converted = self.narrator.text_to_speech_df(batch)
        print(" Done Running text_to_speech_df") 

        # restore order of sentences
        print("restore order of sentences")
        batch_converted.sort_values("index", ascending=True, inplace=True)

        # recombine and save sound
        print("recombine batch")
        waveform = torch.cat(tuple(batch_converted.waveform), dim=1)

        print("saving sound")
        self.narrator.save_audio(case_file + ".wav", waveform)
        print("done saving sound")

        self.chunker.save_chunks_as_text(case_file + ".md", batch)

        # create a report
        print("creating report")
        durations = batch_converted.durations_sec
        #durations_sec = (durations / sampling_freq).tolist()
        perc_sile = 1- sum(durations)/(max(durations)*len(durations))
        
        print("writing results")
        result = {
            "time": tstp,
            "experiment_id": tstp,
            "chunk_durations": list(durations),
            "avg_percent_silence": perc_sile
        }
        result.update(self.case)
        
        print("combining tts_profiler results")
        tts_stage = self.summarize_profile(self.tts_profiler)
        tts_stage.update(result)
        
        print("combining vocoder_profiler results")
        voc_stage = self.summarize_profile(self.vocoder_profiler)
        voc_stage.update(result)

        return [tts_stage, voc_stage]
    

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
        vocoder_model = HIFIGAN.from_hparams(
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
