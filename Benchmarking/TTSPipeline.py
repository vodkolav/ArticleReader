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


from datetime import datetime
import os
import json


class TTSPipeline(Pipeline):
    def __init__(self, output_dir = "output", 
                 checkpoints_dir="checkpoints", 
                 patt = "*"):
        self.output_dir = output_dir
        self.checkpoints_dir = checkpoints_dir
        # self.tele = None
        self.initializers = {
                ".data.test_data": self.init_preprocess,
                ".meta.device": self.init_device,
                ".model_tts.name": self.init_tts_model,
                ".model_voc.name": self.init_voc_model,
                ".meta.chunk_length": self.init_chunker,
                ".meta.batch_size": self.init_batch,
            }
        
        self.delamination_spec = [
            ".tracks.episodes.data", 
            ".tracks.resources.data", 
            ".tracks.log.data"] 
        
        self.current_case = {}
        #self.first = True

    def set_telemetry(self, telem: TelemetryManager):
        level = ""
        pth = level + ".tracks.resources"
        self.tele = telem
        self.run_case = self.tele.add_memory_monitor(self.run_case, {}, pth )


    def init_telemetry(self, new_case):
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
        
        tmp = get_path(pth, new_case)
        if ssr in tmp:
            conf = tmp[ssr]
            lbl = f"{level}.tracks.{ssr}"
            self.narrator.text_to_speech_df = self.tele.add_EpisodeTracker(
                self.narrator.text_to_speech_df, self.narrator.batch_summary, conf, lbl)

        self.narrator.telemetry = self.tele
        self.chunker.telemetry = self.tele
        self.tele.start(new_case)


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

        lim = new_case["meta"].get("limit",None)
        if lim:
            a, b = lim
            self.tele.print(f"limited to chunks {a} to {b}")
            self.chunker.limit = lim



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
        # changing device forces re-initialization of models
        self.init_tts_model(new_case)
        self.init_voc_model(new_case)


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
            templ = json.load(f)
        return templ


    def init_case(self, new_case):
        # TODO: move to base class? 

        if self.current_case == new_case:
            return  # raise Error;  all fields are already identical, which should not happen

        first = False
        if self.current_case == {}:
            self.current_case = new_case
            first = True  # raise Error;  all fields are already identical, which should not happen

        #TODO: check for all parameters in cases, whether they've changed - not just initializers

        for key, init_func in self.initializers.items():
            try:
                cur_val = get_path(key, self.current_case)
                new_val = get_path(key, new_case)
            except KeyError as e:
                raise ValueError(f"Case is missing required key: {key}")

            if cur_val != new_val or first:
                init_func(new_case)
                upd_path(key, self.current_case, new_val)
            else:
                continue  # already initialized to the same value
        first = False


    def run_case(self, new_case):

        tstp = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
        case_file = os.path.join(self.output_dir, tstp)

        # save chunks as markdown for debugging
        self.chunker.save_chunks_as_text(case_file + ".md")

        # TTS
        # sort chunks by len for effeciency
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
        self.narrator.save_audio(case_file + ".wav", waveform)
        self.tele.print("done saving sound")


    def close_case(self, new_case):
        # create a report
        self.tele.print("creating report")
        self.tele.end()
        #result.update(models_result)


    def execute(self, new_case):
        try:
            #TODO: define test batch in new_case.data.[from_chunk, to_chunk ] or something
            #chunks = self.chunker.get_dbg_subset(case["batch_size"], fr)

            self.init_case(new_case)
            self.init_telemetry(new_case)
            self.run_case(new_case)
            self.close_case(new_case)

        except Exception as e:
            print('what')
            return "Error"

        return "Ok"