from ArticleReader.Chunker import Chunker
from ArticleReader.LatexToSpeech import LatexParser
from ArticleReader.Narrator import Narrator
from Benchmarking.MemoryMonitor import MemoryMonitor
from Benchmarking.Pipeline import Pipeline


import torch
from speechbrain.inference import HIFIGAN, Tacotron2


import datetime
import os


class TTSPipeline(Pipeline):
    def __init__(self, output_dir = "output", checkpoints_dir="checkpoints", patt = "*"):
        self.output_dir = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.donecases = self.load_benchmarks(patt)
        self.current_case = {}


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


    def init_batch(self, new_case):

        batch_size = new_case["meta"]["batch_size"]
        # take batches of sorted chunks
        fr = 0 # beginning from chunk
        self.chunks = self.chunker.get_chunks_sorted(batch_size, fr)


    def init_voc_model(self, new_case ):

        voc_model_name = new_case["model_voc"]["name"]
        provider = new_case["model_voc"]["provider"]

        self.vocoder_model = HIFIGAN.from_hparams(
                        source=f"{provider}/{voc_model_name}",
                        savedir=f"{self.checkpoints_dir}/{voc_model_name}",
                        run_opts={"device":self.device}
                        )
        self.vocoder_model.id = voc_model_name

        if new_case["model_voc"]["track"]["memory"]:
            # TODO: decide if attach monitor here or at run
            self.voc_profiler = MemoryMonitor()
            self.vocoder_model.decode_batch = self.tts_profiler.attach_to(self.vocoder_model.decode_batch)


    def init_device(self, new_case):
        self.device = "cuda" if new_case["meta"]["device"] =="GPU" else "cpu"
        # changing device forces re-initialization of models
        self.init_tts_model(new_case)
        self.init_voc_model(new_case)


    def init_tts_model(self, new_case):
        tts_model_name = new_case["model_tts"]["name"]
        provider = new_case["model_tts"]["provider"]
        overrides = new_case["model_tts"].get("overrides", {})

        self.tts_model = Tacotron2.from_hparams(
                    source=f"{provider}/{tts_model_name}",
                    savedir=f"{self.checkpoints_dir}/{tts_model_name}",
                    overrides=overrides,
                    run_opts={"device":self.device}
                    )
        self.tts_model.id = tts_model_name

        if new_case["model_tts"]["track"]["memory"]:
            # TODO: decide if attach monitor here or at run
            self.tts_profiler = MemoryMonitor()
            self.tts_model.encode_batch = self.tts_profiler.attach_to(self.tts_model.encode_batch)


    def init_case(self, new_case):

        if self.current_case == new_case:
            return  # raise Error;  all fields are already identical, which should not happen


        initializers = {
                "data.test_data": self.init_preprocess,
                "meta.device": self.init_device,
                "model_tts": self.init_tts_model,
                "model_voc": self.init_voc_model,
                "meta.chunk_length": self.init_chunker,
                "meta.batch_size": self.init_batch,
            }
        #TODO: turn these into [][] dict indexing

        for key, init_func in initializers.items():
            if key not in new_case:
                raise ValueError(f"Case is missing required key: {key}")

            if self.current_case[key] != new_case[key]:
                init_func(new_case)
                self.current_case["meta.device"] = new_case["meta.device"]
            else:
                continue  # already initialized to the same value


    def run_case(self, new_case):

        #TODO: define test batch in new_case.data.[from_chunk, to_chunk ] or something
        #chunks = self.chunker.get_dbg_subset(case["batch_size"], fr)
        self.init_case(new_case)

        tstp = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
        case_file = os.path.join(self.output_dir, tstp)

        # save chunks as markdown for debugging
        self.chunker.save_chunks_as_text(case_file + ".md")

        # TTS
        self.narrator = Narrator(self.tts_model, self.vocoder_model)
        print(" Running text_to_speech_df")
        data_converted = self.narrator.text_to_speech_df_batched(self.chunker)
        print(" Done Running text_to_speech_df")

        # restore order of sentences
        print("restore order of sentences")
        data_converted.sort_values("index", ascending=True, inplace=True)

        # recombine and save sound
        print("recombine batch")
        waveform = torch.cat(tuple(data_converted.waveform), dim=1)

        print("saving sound")
        self.narrator.save_audio(case_file + ".wav", waveform)
        print("done saving sound")

        #TODO: isolate this into telemetry manager
        # create a report
        print("creating report")
        durations = data_converted.durations_sec
        #durations_sec = (durations / sampling_freq).tolist()
        perc_sile = 1- sum(durations)/(max(durations)*len(durations))

        print("writing results")
        result = {
            "summary":
            {
                "time": tstp,
                "experiment_id": tstp,
                "chunk_durations": list(durations),
                "avg_percent_silence": perc_sile
            }
        }
        result.update(new_case)

        print("combining tts_profiler results")
        tts_stage = self.tts_profiler.summarize_profile()

        print("combining vocoder_profiler results")
        voc_stage = self.voc_profiler.summarize_profile()

        models_result = {
            "model_tts": tts_stage,
            "model_voc": voc_stage
        }

        result.update(models_result)

        return result