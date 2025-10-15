import warnings

# Ignore all FutureWarnings globally
warnings.filterwarnings("ignore", category=FutureWarning)


from Benchmarking.Benchmarking import Bench

from Benchmarking.TTSPipeline import TTSPipeline

ttsppl = TTSPipeline()

case_template = ttsppl.case_template()

#bench = Bench(folder="20251010-1822") # open existing folder

bench = Bench() # create new folder


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
             ".meta.limit": [[20,35]], 
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".meta.device": ["CPU"], 
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"],
             ".data.limit": [[0,6583]]
       }

# order of parameters in the grid is important! 
smallgrid = {
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"],
             ".data.limit": [[0,6583]],
             ".meta.overrides.max_decoder_steps": [1000,2000],
             ".meta.device": ["CPU"], 
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".meta.chunk_length": [100,200],
             #".meta.limit": [[10,20]],
             ".meta.batch_size": [10,20],
       }

medgrid = {  ".meta.chunk_length": [75, 100, 200],
             ".meta.batch_size": [10, 20, 30],
             ".meta.limit": [[20,300]], 
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".meta.device": ["CPU"], 
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"]
}

largegrid = {".meta.chunk_length": [75, 100, 200, 300, 500, 800],
             ".meta.batch_size": [10, 20, 30, 50],
             ".meta.limit": [[0,300]], 
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".meta.device": ["CPU"], 
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"]
}

chosengrid =  smallgrid  # twogrid  # medgrid  # onegrid # 

bench.unfurl_grid(case_template, chosengrid)

bench.run_experiments()
