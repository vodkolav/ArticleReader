# 3. Data Processing Functions (processing.py)

# These functions apply transformations regardless of batch or streaming mode.

#import findspark
import os
import sys
import pandas as pd 
import torch

from pyspark.sql.functions import pandas_udf, PandasUDFType, udf, col, lit, desc, floor

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark import pandas as psp

from ArticleReader.Chunker import Chunker
from ArticleReader.LatexToSpeech import LatexParser
from ArticleReader.Narrator import Narrator

from pyspark.sql import functions as F

def compute_waveform_lengths(df):
    return df.withColumn("waveform_length", F.size("waveform"))

def concatenate_waveforms(df, order_column):
    df_ordered = df.orderBy(order_column)
    return df_ordered.agg(F.flatten(F.collect_list("waveform")).alias("concatenated_waveform"))

@pandas_udf(StructType([StructField("waveform", ArrayType(FloatType())),StructField("mel_lengths", IntegerType())]))
def predict_batch_udf(sentences: pd.Series) -> pd.DataFrame:
  # TODO: calculate and store "seq_len"
  # TODO: non-default model initialization
  narrator = Narrator()    
  waveforms, mel_lengths = narrator.infer(sentences)

  arr = torch.tensor_split(waveforms.squeeze(1), len(waveforms), dim=0)

  # Add more pause where needed (very naive currenty)
  mel_lengths = narrator.add_pauses(sentences, mel_lengths, pause_dur=40)   
  # Cut silence padding while applying pauses from above 
  arr = [a[:, :l].squeeze(0).numpy() for a, l in zip(arr, mel_lengths * narrator.hop_len)]  
  
  output = pd.DataFrame({"waveform": arr, "mel_lengths": mel_lengths})
 
  return output

def output_sound(wfc):
    waveform = wfc.collect()[0]
    narrator = Narrator()
    tens = torch.Tensor(waveform)
    case_file = "output/spark_test"
    print("saving sound")
    narrator.save_audio(case_file + ".wav",tens )
    print("done saving sound")


def preprocess_text():

    parser = LatexParser()
    content = parser.read_latex(input_file)
    processed = parser.custom_latex_to_text(content)
    parser.save_text(processed, "dbg/spec_my.txt")

    tables = parser.get_tables()
    parser.save_text(tables, "dbg/tables.tex")

def chunk_text():
    chunker = Chunker(max_len=200)
    chunker.split_text_into_chunks(processed)
    chunks = chunker.get_test_batch(10, 0)
    # chunks = chunker.chunks
    chunker.save_chunks_as_text(output_file + ".md", chunks)
    print("text chunks:", [len(ch) for ch in chunks])