import warnings

# Ignore all FutureWarnings globally
warnings.filterwarnings("ignore", category=FutureWarning)


from Benchmarking.Benchmarking import Bench
from Benchmarking.utils import isDebugging
from Benchmarking.TTSPipeline import TTSPipeline


onegrid = {  
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"],
             ".meta.device": ["CPU"], 
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".meta.chunk_length": [75],
             ".meta.batch_size": [2],
             "meta.chunks_limit": [[20,30]], # for range parameters, to prevent grid-expansion,
                                             # make them a single-item array 
       }

twogrid = {  
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"],
             ".data.limit": [[0,6583]],
             ".meta.device": ["CPU"], 
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".meta.chunk_length": [75],
             ".meta.chunks_limit": [[20,35]], 
             ".meta.batch_size": [2,3],
       }

# order of parameters in the grid is important! 
#TODO:make it match the order in TTSpipeline.initializers ?
# or warn that it does not ? 
smallgrid = {
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"],
             ".data.limit": [[0,6583]],
             ".meta.device": ["CPU"], 
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".model_tts.overrides.max_decoder_steps": [500],
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".meta.chunk_length": [100, 200],
             ".meta.chunks_limit": [[20,30 ]],
             ".meta.batch_size": [5],
       }


small_heavy_grid = {
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"],
             ".data.limit": [[None,None]],
             ".meta.device": ["CPU"], 
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".model_tts.overrides.max_decoder_steps": [3000],
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".meta.chunk_length": [500,800],
             ".meta.chunks_limit": [[None, None]],
             ".meta.batch_size": [100],
       }

medgrid = {  
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"],
             ".data.limit": [[0,6583]],
             ".meta.device": ["CPU"],                 
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".model_tts.overrides.max_decoder_steps": [500],
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".meta.chunk_length": [75, 100, 200],
             ".meta.chunks_limit": [[20,50]],          
             ".meta.batch_size": [10, 20, 30],             
}

largegrid = {
             ".data.test_data": ["data/arXiv-2106.04624v1/main.tex"],
             ".data.limit": [[0,6583]],
             ".meta.device": ["CPU"],              
             ".model_voc.name": ["tts-hifigan-ljspeech"],
             ".model_tts.overrides.max_decoder_steps": [500, 1000, 2000, 3000],
             ".model_tts.name": ["tts-tacotron2-ljspeech"],
             ".meta.chunk_length": [75, 100, 200, 300, 500, 800, 1000, 1500],
             ".meta.chunks_limit": [[None,None]],
             ".meta.batch_size": [10, 20, 30, 50, 70, 100, 150, 175, 200, 300],
       }


chosengrid = smallgrid #small_heavy_grid #  largegrid  #  medgrid #  twogrid    # onegrid # 
       

if isDebugging(): #debug
       if chosengrid == largegrid:
              print("Nope, I won't run large grid in debug mode!")
              exit()
       else:
              bench = Bench() # create new folder
else: #NOdebug
       if chosengrid == largegrid:
              bench = Bench(folder="20251022/0200") # open existing folder
       else:
              print(" Ithink you want to debug with a smallgrid ")


ttsppl = TTSPipeline()

case_template = ttsppl.case_template()
#case_template["tracks"].pop('profile')
bench.configure(ttsppl)

bench.unfurl_grid(case_template, chosengrid)

bench.run_experiments()
