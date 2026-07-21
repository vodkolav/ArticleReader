import os
# decrease log verbosity of speechbrain
os.environ["SB_LOG_LEVEL"] = "50"

from ArticleReader.Chunker import Chunker
from ArticleReader.LatexToSpeech import LatexParser
from ArticleReader.Narrator import Narrator

from Benchmarking.Pipeline import Pipeline
from Benchmarking.utils import get_path, upd_path
from Benchmarking.telemetry_manager import TelemetryManager

import torch
from speechbrain.inference import HIFIGAN, Tacotron2

import os
import json


class TTSPipeline(Pipeline):
    def __init__(self, checkpoints_dir="checkpoints", 
                 patt = "*"):
        
        self.checkpoints_dir = checkpoints_dir
        # self.tele = None

        # Order of initializers matters, as some depend on others.
        # If a parameter changes, all downstream initializers must re-run.
        # * since Python 3.7 dicts preserve insertion order
        self.initializers = {
                ".data.test_data": self.init_preprocess,            
                ".meta.device": self.init_device,
                ".model_voc.name": self.init_voc_model,
                ".model_tts.overrides.max_decoder_steps": self.init_overrides,
                ".model_tts.name": self.init_tts_model,
                ".meta.chunk_length": self.init_chunker,
                ".meta.chunks_limit": self.init_limit,
                ".meta.batch_size": self.init_batch,
            }
        
        self.delamination_spec = [
            ".tracks.episodes.data", 
            ".tracks.resources.data", 
            ".tracks.log.data"] 
        
        self.current_case = {}



    def set_telemetry(self, tele: TelemetryManager):
        self.tele = tele
        self.tele.intercept_logs("speechbrain")
        # The MemoryMonitor tracks the memory usage of the whole process.
        # It is reset on execution of each case
        # TODO: Then how do we measure mem usage of individual components?


        #disable profiler temporarily    
        # pth = ".tracks.profile"
        # conf = get_path(pth, self.case_template())
        # self.tele.AttachSensor(self, "init_case", pth, config = conf)
        # self.tele.AttachSensor(self, "run_case", pth, config = conf)
        # self.tele.AttachSensor(self, "execute", pth, config = conf)


        pth = ".tracks.resources"
        conf = get_path(pth, self.case_template())
        # at this point case is not yet initialized, so we use default config from case_template
        self.tele.AttachSensor(self, "execute", pth, config = conf)


    def init_telemetry(self):
        # re-runs for every new case
        # TODO: attach monitors for particular pipeline components 

        tracks = [
            ".tracks.episodes",
            #".tracks.profile"
            #,".tracks.resources" 
            #,".tracks.log",
            #,".model_tts.tracks.log"
            ] 

        # TODO: interesting idea: maybe we can use the AttachSensor routine
        #  to attach to builtin python logger

        pth = tracks[0]
        self.tele.AttachSensor(self.narrator, "text_to_speech_df", pth, summary_func = "batch_summary")


    def init_preprocess(self, new_case):

        input_file = new_case["data"]["test_data"]
        data_limit = new_case["data"].get("limit",{})
        self.tele.print("parsing LaTeX to narratable text... why its taking so long?")
        parser = LatexParser()
        content = parser.read_latex(input_file)
        self.preprocessed_text = parser.custom_latex_to_text(content)
        if data_limit:
            self.preprocessed_text = self.preprocessed_text[slice(*data_limit)]
        self.tables = parser.get_tables()

            # save debug info
        # tstp = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
        # dbg_dir = "dbg/" + tstp
        parser.save_text(self.preprocessed_text, self.case_file + ".txt")
        # parser.save_text(self.tables, "dbg/tables.tex")


    def init_chunker(self, new_case):

        chunk_length = new_case["meta"]["chunk_length"]

        self.chunker = Chunker(max_len=chunk_length)
        self.chunker.telemetry = self.tele
        self.chunker.split_text_into_chunks(self.preprocessed_text)


    def init_limit(self, new_case):
        lim = new_case["meta"].get("chunks_limit",None)
        if lim:
            a, b = lim
            self.tele.print(f"limited to chunks {a} to {b}")

        self.chunker.limit_chunks(lim)


    def init_batch(self, new_case):

        self.chunker.batch_size = new_case["meta"]["batch_size"]


        self.narrator = Narrator(self.tts_model, self.vocoder_model)
        self.narrator.tele = self.tele
        # TODO implement: 
        #fr = 0 # beginning from chunk

        #self.chunks = self.chunker.get_chunks_sorted(batch_size, fr)

    def init_overrides(self, new_case):
        #the only override used currently is max_decoder_steps
        #and it is used during tts_model init.
        #still, this function is required  to trigger the change of this parameter 
        #TODO: validate that it runs BEFORE init_tts_model
        print("oh really?")
        pass

    def init_voc_model(self, new_case ):
        model = new_case["model_voc"]
        voc_model_name = model["name"]
        provider = model["provider"]

        self.vocoder_model = HIFIGAN.from_hparams(
                        source=f"{provider}/{voc_model_name}",
                        savedir=f"{self.checkpoints_dir}/{voc_model_name}",
                        run_opts={"device":self.device}
                        )
        #self.vocoder_model.id = voc_model_name


    def init_device(self, new_case):
        self.device = "cuda" if new_case["meta"]["device"] =="GPU" else "cpu"


    def init_tts_model(self, new_case):
        model = new_case["model_tts"]
        tts_model_name = model["name"]
        provider = model["provider"]
        overrides = model.get("overrides", None)

        self.tts_model = Tacotron2.from_hparams(
                    source=f"{provider}/{tts_model_name}",
                    savedir=f"{self.checkpoints_dir}/{tts_model_name}",
                    overrides=overrides,
                    run_opts={"device":self.device}
                    )
        
        # TODO: still need this?
        #self.tts_model.id = tts_model_name


    def case_template(self):
        # /home/michael/Projects/ArticleReader/
        #TODO:add more settings:
        # timestamp formats
        # timezone of all times
        
        fl = "Benchmarking/config/ArticleReader.json"
        with open(fl, 'r') as f:
            template = json.load(f)
        return template



    @property
    def case_file(self):
        return self.tele.case_filename()


    def run_case(self):

        # save chunks as markdown for debugging
        self.chunker.save_chunks_as_text(self.case_file + ".md")

        # TTS
        # sort chunks by len for efficiency
        self.chunker.sort_by_text_len()

        self.tele.print(" Running text_to_speech_df")
        data_converted = self.narrator.text_to_speech_df_batched(
            self.chunker.feed_df_batches())
        self.tele.print(" Done Running text_to_speech_df")

        # restore order of sentences
        self.tele.print("restore order of sentences")
        data_converted.sort_values("index", ascending=True, inplace=True)

        # recombine and save sound
        self.tele.print("recombine batch")
        waveform = torch.cat(tuple(data_converted.waveform), dim=1)

        self.tele.print("saving sound")
        self.narrator.save_audio(self.case_file + ".wav", waveform)
        self.tele.print("done saving sound")


    def results(self):
        return self.tele.results()