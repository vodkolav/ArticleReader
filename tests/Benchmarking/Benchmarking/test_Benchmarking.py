from Benchmarking.Benchmarking import Bench

from Benchmarking.TTSPipeline import TTSPipeline

ttsppl = TTSPipeline()

case_template = ttsppl.case_template()

bench = Bench(folder="20251009-1501") # open existing folder

#bench = Bench() # create new folder


bench.configure(ttsppl)

onegrid = {  ".meta.chunk_length": [75],
             ".meta.batch_size": [2],
            # "meta.limit": [20,30], probably will try to grid-expand it 
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".meta.device": ["CPU"], 
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"]
       }

twogrid = {  ".meta.chunk_length": [75],
             ".meta.batch_size": [2,3],
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".meta.device": ["CPU"], 
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"]
       }

smallgrid = {".meta.chunk_length": [75,100],
             ".meta.batch_size": (2, 3),
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".meta.device": ["CPU"], 
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"]
       }
smallgrid

pathspec = [".tracks.episodes.data", 
            ".tracks.episodes.run_time_sec",
            ".tracks.episodes.exceptions",
            ".tracks.resources.data", 
            ".tracks.log.data"] 

chosengrid = onegrid
bench.unfurl_grid(case_template, chosengrid, pathspec)

print("len(bench.TODOcases):", len(bench.TODOcases))

#bench.TODOcases
bench.run_experiments()


print("bench press done")