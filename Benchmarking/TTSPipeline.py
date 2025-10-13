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

import time
from datetime import datetime
import os
import json


class TTSPipeline(Pipeline):
    def __init__(self, checkpoints_dir="checkpoints", 
                 patt = "*"):
        
        self.checkpoints_dir = checkpoints_dir
        # self.tele = None
        self.tstp_format = "%Y%m%d-%H%M%S"

        # Order of initializers matters, as some depend on others.
        # If a parameter changes, all downstream initializers must re-run.
        # * since Python 3.7 dicts preserve insertion order
        self.initializers = {
                ".meta.device": self.init_device,
                ".model_tts.name": self.init_tts_model,
                ".model_voc.name": self.init_voc_model,
                ".data.test_data": self.init_preprocess,
                ".meta.chunk_length": self.init_chunker,
                ".meta.limit": self.init_limit,
                ".meta.batch_size": self.init_batch,
            }
        
        self.delamination_spec = [
            ".tracks.episodes.data", 
            ".tracks.resources.data", 
            ".tracks.log.data"] 
        
        #self.first = True


    @property
    def current_case(self):
        return self.tele.case

    @current_case.setter
    def current_case(self,val):
        self.tele.case = val


    def set_telemetry(self, tele: TelemetryManager):
        level = ""
        pth = level + ".tracks.resources"
        self.tele = tele
        self.execute = self.tele.add_memory_monitor(self.execute, {}, pth )


    def init_telemetry(self):
        # re-runs for every new case

        # self.tele.__init__() #? 
        # TODO: implement addressing tracks of particular pipeline components 
        # through jq path, eg: .model_tts.tracks.resources

        #if new_case["tracks"]["resources"]:
            # TODO: attach monitors for particular pipeline components 
        #    self.run_case = self.tele.add_memory_monitor(self.run_case, "" )

        # if new_case["tracks"]["log"]:
        #     self.tele.enable_logging()


        # model = new_case["model_voc"]
        # if model["tracks"]["resources"]:
        #     # TODO: decide if attach monitor here or at run
                # actually I don't need to attach this to different models, as the monitor just tracks
                # the memory usage of the whole process.
        #     self.vocoder_model.decode_batch = self.tele.add_memory_monitor(self.vocoder_model.decode_batch, "model_voc")
        level = ""

        ssr = "episodes"
        pth = level + ".tracks"
        
        tmp = get_path(pth, self.current_case)
        if ssr in tmp:
            conf = tmp[ssr]
            lbl = f"{level}.tracks.{ssr}"
            self.narrator.text_to_speech_df = self.tele.add_EpisodeTracker(
                self.narrator.text_to_speech_df, self.narrator.batch_summary, conf, lbl)

        self.narrator.telemetry = self.tele
        self.chunker.telemetry = self.tele


    def init_preprocess(self, new_case):

        input_file = new_case["data"]["test_data"]

        parser = LatexParser()
        content = parser.read_latex(input_file)
        self.preprocessed_text = parser.custom_latex_to_text(content)
        self.tables = parser.get_tables()


            # save debug info
        # tstp = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
        # dbg_dir = "dbg/" + tstp
        # parser.save_text(self.preprocessed_text, "dbg/postprocessed.txt")
        # parser.save_text(self.tables, "dbg/tables.tex")


    def init_chunker(self, new_case):

        chunk_length = new_case["meta"]["chunk_length"]

        self.chunker = Chunker(max_len=chunk_length)
        self.chunker.split_text_into_chunks(self.preprocessed_text)


    def init_limit(self, new_case):
        lim = new_case["meta"].get("limit",None)
        if lim:
            a, b = lim
            self.tele.print(f"limited to chunks {a} to {b}")

        self.chunker.limit_chunks(lim)


    def init_batch(self, new_case):

        self.chunker.batch_size = new_case["meta"]["batch_size"]


        self.narrator = Narrator(self.tts_model, self.vocoder_model)
                
        # TODO implement: 
        #fr = 0 # beginning from chunk

        #self.chunks = self.chunker.get_chunks_sorted(batch_size, fr)


    def init_voc_model(self, new_case ):
        model = new_case["model_voc"]
        voc_model_name = model["name"]
        provider = model["provider"]

        self.vocoder_model = HIFIGAN.from_hparams(
                        source=f"{provider}/{voc_model_name}",
                        savedir=f"{self.checkpoints_dir}/{voc_model_name}",
                        run_opts={"device":self.device}
                        )
        self.vocoder_model.id = voc_model_name


    def init_device(self, new_case):
        self.device = "cuda" if new_case["meta"]["device"] =="GPU" else "cpu"


    def init_tts_model(self, new_case):
        model = new_case["model_tts"]
        tts_model_name = model["name"]
        provider = model["provider"]
        overrides = model.get("overrides", {})

        self.tts_model = Tacotron2.from_hparams(
                    source=f"{provider}/{tts_model_name}",
                    savedir=f"{self.checkpoints_dir}/{tts_model_name}",
                    overrides=overrides,
                    run_opts={"device":self.device}
                    )
        
        # TODO: still need this?
        self.tts_model.id = tts_model_name


    def case_template(self):
        # /home/michael/Projects/ArticleReader/
        fl = "Benchmarking/config/ArticleReader.json"
        with open(fl, 'r') as f:
            template = json.load(f)
        return template


    def init_case(self, new_case):
        # TODO: move to base class? 

        force = self.tele.start(new_case) # or new_case? 

        #TODO: check for all parameters in cases, whether they've changed - not just initializers

        for key, init_func in self.initializers.items():
            try:
                cur_val = get_path(key, self.current_case)
                new_val = get_path(key, new_case)
            except KeyError as e:
                raise ValueError(f"Case is missing required key: {key}")

            different = cur_val != new_val
            if different or force:
                force = True # once a change is detected, all downstream initializers must run
                if not different:
                    self.tele.print(f" initializing {key} to {new_val}")
                else:
                    self.tele.print(f" re-initializing {key} from {cur_val} to {new_val}")
                init_func(new_case)
                self.current_case = upd_path(key, self.current_case, new_val)
            else:
                continue  # already initialized to the same value
        force = False


    def now(self):
        # TODO: variable format
        return time.time()


    def timestamp(self, entry = None):
        if entry:
            return datetime.fromtimestamp(entry)\
                           .strftime(self.tstp_format)  #TODO:  should be in config
        else:
            return self.timestamp(self.now())

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


    def close_case(self):
        self.tele.end()
        #result.update(models_result)


    def execute(self, new_case):
        try:
            #TODO: define test batch in new_case.data.[from_chunk, to_chunk ] or something
            #chunks = self.chunker.get_dbg_subset(case["batch_size"], fr)

            self.init_case(new_case)
            self.init_telemetry()
            self.run_case()
            self.close_case()

        except Exception as e:
            print('what')
            return "Error"

        return "Ok"