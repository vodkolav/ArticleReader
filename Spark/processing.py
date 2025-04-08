# Data Processing Functions (processing.py)

# These functions apply transformations regardless of batch or streaming mode.
import numpy as np
import pandas as pd 
import torch

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, FloatType, StringType, TimestampType

from ArticleReader.Chunker import Chunker
from ArticleReader.LatexToSpeech import LatexParser
from ArticleReader.Narrator import Narrator

import soundfile as sf
from datetime import timedelta
import config as conf


tts_schema = StructType([
    StructField("waveform", ArrayType(FloatType())),
    StructField("mel_lengths", IntegerType()),
    StructField("seq_len", IntegerType()),
    StructField("duration", FloatType())])

@pandas_udf(tts_schema)
def predict_batch_udf(sentences: pd.Series) -> pd.DataFrame:
  # TODO: calculate and store "seq_len"
  # TODO: non-default model initialization
  # TODO: also, for supporting streaming, the models somehow 
  # have to be persistent between requests
  narrator = Narrator()  

  # ensure sentences are sorted by seq_len
  batch_df = sentences.to_frame("sentences")

  batch_df.reset_index(inplace=True)  
  batch_df.loc[:,"seq_len"] = batch_df.sentences.map(narrator.seq_len)
  batch_df.sort_values("seq_len", ascending=False, inplace=True)
  
  batch_df.head(15)

  waveforms, mel_lengths = narrator.infer(batch_df.sentences)

  arr = torch.tensor_split(waveforms.squeeze(1), len(waveforms), dim=0)

  # Add more pause where needed (very naive currenty)
  batch_df["mel_lengths"] = narrator.add_pauses(batch_df.sentences, mel_lengths, pause_dur=40)   

  batch_df["duration"] = mel_lengths * narrator.hop_len / 22050.0

  # Cut silence padding while applying pauses from above 
  batch_df["waveform"] = [a[:, :l].squeeze(0).numpy() for a, l in zip(arr, mel_lengths * narrator.hop_len)]  
  
  batch_df.sort_values("index", inplace=True)

  output = batch_df[["waveform", "mel_lengths", "seq_len", "duration"]]
  return output


def save_to_disk(wav_data, file_path):

    """ Function to save speech data as a WAV file """
    sr = 22050
    
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
    # For example, write a file named using the request_id and timestamp.
    output_path = conf.output_path + f"/{request_id}"
    save_to_disk(speech, output_path + ".wav")
    save_srt(row["text"],  output_path + ".srt")
    save_video(output_path)


def save_srt(content, filename):
    with open(filename, "w+") as f:
        f.write(content)

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
    
    return chunks 

def udf_split_text(text, chunk_size=500, test =False):
    chunker = Chunker(max_len=chunk_size)
    #print(text)
    chunker.split_text_into_chunks(text)     
    return chunker.get_chunks()


window_schema = StructType([
    StructField("start", TimestampType(), True),
    StructField("end",   TimestampType(), True)
])


processed_schema = StructType([
        StructField("window", window_schema, False),
        StructField("request_id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("tables", ArrayType(visuals_schema), True),
        StructField("figures", ArrayType(visuals_schema), True)   
    ])


# Define the output schema
recomb_schema = StructType([
    StructField("request_id", StringType()),
    StructField("speech", ArrayType(FloatType())),
    StructField("text", StringType()),
])

# Grouped applyInPandas function
def concat_waveforms_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf["index"] = pd.to_numeric(pdf["index"], errors="raise")
    pdf_sorted = pdf.sort_values("index")
    print(type(pdf_sorted["waveform"].iloc[0]))
    concatenated = np.concatenate(pdf_sorted["waveform"].values).tolist()
    conc_text = generate_srt(pdf_sorted["sentence"].array, pdf_sorted["duration"].array)
    return pd.DataFrame({
        "request_id": [pdf["request_id"].iloc[0]],
        "speech": [concatenated],
        "text": [conc_text]
    })



def generate_srt(sentences, durations):
    """
    Generates an SRT file from a list of sentences and durations.

    Args:
        sentences (list of str): The sentences to be shown as subtitles.
        durations (list of int): Durations (in seconds) for each sentence in chronological order.
        output_file (str): Path to save the SRT file.

    Returns:
        None
    """

    def format_timestamp(seconds):
        """Formats seconds into SRT timestamp format."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        milliseconds = int(td.microseconds / 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    srt_content = ""

    # md = min(durations)
#    duration = [ a - md for a in duration]

    current_time = 0  # Start time in seconds
    for i, (sentence, duration) in enumerate(zip(sentences, durations), start=1):
        duration = float(duration)
        start_time = format_timestamp(current_time)
        end_time = format_timestamp(current_time + duration)
        current_time += duration  # Increment time by the duration of the sentence

        # Write SRT entry
        srt_content += f"{i}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{sentence.strip()}\n\n"

    return srt_content

def save_video(output_file):
    # ffmpeg -loop 1 -i image.png -i 24.12.08-17.wav -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest output.mp4

    import os

    print("\n saving static video")
    command = f"""
ffmpeg -y -loop 1 -i output/image.png -i {output_file}.wav 
-c:v libx264 -tune stillimage -c:a aac -b:a 
192k -pix_fmt yuv420p -shortest {output_file}.mp4
"""
    command = command.replace("\n", "")
    print(command)
    os.system(command)

# Declare a grouped aggregate pandas UDF
@pandas_udf(ArrayType(FloatType()))
def concat_waveforms(df: pd.DataFrame) -> pd.Series:
    # Sort by index
    df_sorted = df.sort_values("index")
    # Concatenate all waveform arrays
    concatenated = np.concatenate(df_sorted["waveform"].values).tolist()
    return pd.Series([concatenated])
