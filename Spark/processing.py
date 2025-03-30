# 3. Data Processing Functions (processing.py)

# These functions apply transformations regardless of batch or streaming mode.

#import findspark
import os
import sys
import pandas as pd 
import torch

from pyspark.sql.functions import pandas_udf, PandasUDFType, udf, col, lit, desc, floor

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, FloatType, StringType, TimestampType

from pyspark import pandas as psp

from ArticleReader.Chunker import Chunker
from ArticleReader.LatexToSpeech import LatexParser
from ArticleReader.Narrator import Narrator

import math

import soundfile as sf

import config as conf

def compute_waveform_lengths(df):
    return df.withColumn("waveform_length", F.size("waveform"))

def concatenate_waveforms(df, order_column):
    df_ordered = df.orderBy(order_column)
    return df_ordered.agg(F.flatten(F.collect_list("waveform")).alias("concatenated_waveform"))

tts_schema = StructType([
    StructField("waveform", ArrayType(FloatType())),
    StructField("mel_lengths", IntegerType()),
    StructField("seq_len", IntegerType())])

@pandas_udf(tts_schema)
def predict_batch_udf(sentences: pd.Series) -> pd.DataFrame:
  # TODO: calculate and store "seq_len"
  # TODO: non-default model initialization
  # TODO: also, for supporting streaming, the models somehow 
  # have to be persistent between requests
  narrator = Narrator()  

  # ensure sentences are sorted by seq_len
  batch_df = sentences.to_frame("sentences")

  batch_df.loc[:,"seq_len"] = batch_df.sentences.map(narrator.seq_len)
  batch_df.sort_values("seq_len", ascending=False, inplace=True)
  
  batch_df.head(15)

  waveforms, mel_lengths = narrator.infer(batch_df.sentences)

  arr = torch.tensor_split(waveforms.squeeze(1), len(waveforms), dim=0)

  # Add more pause where needed (very naive currenty)
  mel_lengths = narrator.add_pauses(batch_df.sentences, mel_lengths, pause_dur=40)   
  # Cut silence padding while applying pauses from above 
  arr = [a[:, :l].squeeze(0).numpy() for a, l in zip(arr, mel_lengths * narrator.hop_len)]  
  
  output = pd.DataFrame({"waveform": arr, "mel_lengths": mel_lengths, "seq_len": batch_df.seq_len })
 
  return output

# A UDF to concatenate the sorted waveforms
@pandas_udf(ArrayType(FloatType()))
def concat_waveforms(sorted_waveforms):
    # sorted_waveforms is a list of structs with "index" and "waveform"
    return b"".join([row for row in sorted_waveforms])


def save_to_disk(wav_data, file_path):

    """ Function to save speech data as a WAV file """
    sr = 22050
    # print("row shape: ", row.shape)
    # file_path = conf.output_path + str(row.timestamp) + ".wav"
    # wav_data = row.speech  # Assuming speech is a NumPy array or bytes
    
    # Save using soundfile (if NumPy array) or write raw bytes
    print("saving sound to ", file_path)
    if isinstance(wav_data, bytes):
        with open(file_path, "wb") as f:
            f.write(wav_data)
    else:
        sf.write(file_path, wav_data, samplerate=sr)
    print("done saving sound")

# Define your pandas function that writes the row to disk.
def write_row(row):
    # Loop over rows and write to disk using your custom logic.
    # Implement your custom write logic here, e.g.:
    request_id = row["request_id"]
    speech = row["speech"]
    timestamp = row["timestamp"]
    # For example, write a file named using the request_id and timestamp.
    output_path = conf.output_path + f"/{request_id}.wav"
    save_to_disk(speech, output_path)



# Define your pandas function that writes the row to disk.
def write_request(pdf: pd.DataFrame) -> pd.DataFrame:
    # Loop over rows and write to disk using your custom logic.
    for _, row in pdf.iterrows():
        # Implement your custom write logic here, e.g.:
        request_id = row["request_id"]
        speech = row["speech"]
        timestamp = row["timestamp"]
        # For example, write a file named using the request_id and timestamp.
        output_path = conf.output_path + f"/{request_id}_{timestamp}.wav"
        save_to_disk(speech, output_path)
    # Optionally, return the input PDF or an empty DataFrame.
    return pdf  # or pd.DataFrame([], columns=pdf.columns)

def custom_write_function(batchDF, batchId):
    # Example: Group by request_id (if not already unique) and apply your custom pandas_udf.
    # Let's assume your custom pandas_udf is called "write_request_udf" and it writes the row to disk,
    # then returns the same row (or some status).
    #
    # NOTE: applyInPandas requires a defined output schema. For example, here we assume the output schema
    # is identical to the input schema, but you could change that as needed.
    
    # If your logic is purely side-effecting (writing to disk) and you don't need to return any data,
    # you might just call a custom function that converts the batch to Pandas and writes each row.
    # For demonstration, here's an approach using applyInPandas:
    
    # from pyspark.sql.types import StructType, StructField, StringType, BinaryType, TimestampType
    # import pandas as pd
    # if batchDF.rdd.isEmpty():
    #     return
    # Define the output schema (here, same as input but you can customize)
    output_schema = batchDF.schema    
    # Apply your custom pandas_udf (as a grouped transformation if needed).
    # If each request is a single row in batchDF, you might not need grouping.
    processed = batchDF.groupBy("request_id").applyInPandas(write_request, schema=output_schema)
    # Optionally, log the number of requests written:
    #print(f"Batch {batchId}: Processed {processed.count()} requests.")



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
    chunks = text_series.apply(udf_split_text, chunk_size=conf.chunk_size)  # Apply function row-wise
    
    return chunks # cum_text_volume(chunks)

def udf_split_text(text, chunk_size=500, test =False):
    from ArticleReader.Chunker import Chunker
    chunker = Chunker(max_len=chunk_size)
    #print(text)
    chunker.split_text_into_chunks(text)     
    return chunker.get_chunks()
    # chunks = chunker.get_all_chronological()
    # return cum_text_volume(chunks)

# Example schema for chunks
# sentences_schema = ArrayType(StructType([
#     # StructField("timestamp", TimestampType(), True),
#     # StructField("request_id", StringType(), True),

#     StructField("text_len", IntegerType(), True),
    
# ]))


window_schema = StructType([
    StructField("start", TimestampType(), True),
    StructField("end",   TimestampType(), True)
])

sentences_schema = StructType([
    #StructField("window", window_schema, False),
    StructField("timestamp", TimestampType(), False),
    StructField("request_id", StringType(), True),
    StructField("index", IntegerType(), True),
    StructField("sentence", StringType(), True),    
    StructField("text_len", IntegerType(), True),
    StructField("cum_text_volume", IntegerType(), True),
])


processed_schema = StructType([
        StructField("window", window_schema, False),
        StructField("request_id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("tables", ArrayType(visuals_schema), True),
        StructField("figures", ArrayType(visuals_schema), True)   
    ])

# def compute_cumsum(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.sort_values(["request_id", "text_len"], ascending=[True, False])
#     df["cum_text_volume"] = df.groupby("request_id")["text_len"].cumsum()
#     return df


def cum_text_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["request_id", "text_len"], ascending=[True, False])  # Sort within request
    df["cum_text_volume"] = df.groupby("request_id")["text_len"].cumsum()  # Return only the cumulative sum
    return df

# def cum_text_volume(pdf):
#     # pdf.reset_index()
#     # print("="*50)
#     # print(type(pdf))
#     # print(pdf.columns)
#     # Sort the chunks within the request if needed (e.g., by text_len or a sequence column)    
#     pdf = pdf.sort_values("text_len")
#     pdf["cum_text_volume"] = pdf["text_len"].cumsum()
#     #.apply(lambda s: math.floor(s / conf.text_volume_max))
#     return pdf


def chunk_text():
    chunker = Chunker(max_len=200)
    chunker.split_text_into_chunks(processed)
    chunks = chunker.get_test_batch(10, 0)
    # chunks = chunker.chunks
    chunker.save_chunks_as_text(output_file + ".md", chunks)
    print("text chunks:", [len(ch) for ch in chunks])
