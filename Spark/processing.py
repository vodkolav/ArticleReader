# 3. Data Processing Functions (processing.py)

# These functions apply transformations regardless of batch or streaming mode.

#import findspark
import os
import sys
import pandas as pd 
import torch

from pyspark.sql.functions import pandas_udf, PandasUDFType, udf, col, lit, desc, floor

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, FloatType, StringType
from pyspark import pandas as psp

from ArticleReader.Chunker import Chunker
from ArticleReader.LatexToSpeech import LatexParser
from ArticleReader.Narrator import Narrator

from pyspark.sql import functions as F

import soundfile as sf

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


def save_to_disk(row):

    """ Function to save speech data as a WAV file """
    output_path="output/"
    sr = 22050
    file_path = output_path + row.request_id + ".wav"
    wav_data = row.speech  # Assuming speech is a NumPy array or bytes
    
    # Save using soundfile (if NumPy array) or write raw bytes
    print("saving sound")
    if isinstance(wav_data, bytes):
        with open(file_path, "wb") as f:
            f.write(wav_data)
    else:
        sf.write(file_path, wav_data, samplerate=sr)
    print("done saving sound")


def output_sound(wfc, output_file):
    waveform = wfc.collect()[0]
    narrator = Narrator()
    tens = torch.Tensor(waveform)
    print("saving sound")
    narrator.save_audio(output_file + ".wav",tens )
    print("done saving sound")


def preprocess_text(file_content):
    parser = LatexParser()
    processed_text = parser.custom_latex_to_text(file_content)
    return processed_text, parser.tables, parser.figures

visuals_schema = StructType([
    StructField("label", StringType(), True),
    StructField("content", StringType(), True)   
])

# Define schema for the UDF return type
preprocess_schema = StructType([
    StructField("text", StringType(), True),
    StructField("tables", ArrayType(visuals_schema), True),
    StructField("figures", ArrayType(visuals_schema), True)
])

@pandas_udf(preprocess_schema)
def preprocess_text_udf(content_series: pd.Series) -> pd.DataFrame:
    return content_series.apply(lambda x: pd.Series(preprocess_text(x)))  # Apply function


# Define schema for chunks
chunk_schema = ArrayType(StringType())

@pandas_udf(chunk_schema)
def split_text_into_chunks_udf(text_series: pd.Series) -> pd.DataFrame:
    return text_series.apply(udf_split_text)  # Apply function row-wise


def udf_split_text(text, chunk_size=500, test =False):
    from ArticleReader.Chunker import Chunker
    chunker = Chunker(max_len=chunk_size)
    #print(text)
    chunker.split_text_into_chunks(text)
    return chunker.get_chunks()

def chunk_text():
    chunker = Chunker(max_len=200)
    chunker.split_text_into_chunks(processed)
    chunks = chunker.get_test_batch(10, 0)
    # chunks = chunker.chunks
    chunker.save_chunks_as_text(output_file + ".md", chunks)
    print("text chunks:", [len(ch) for ch in chunks])
