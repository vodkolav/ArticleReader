# 4. Pipeline Logic (pipeline.py)

# This handles both batch and streaming.

from config import get_spark_session
from processing import compute_waveform_lengths, concatenate_waveforms, output_sound
from processing import  preprocess_text, udf_split_text, predict_batch_udf 
from pyspark.sql import functions as F, Row
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import pandas_udf, PandasUDFType, udf, col, lit, desc, floor, monotonically_increasing_id
from pyspark.sql.functions import collect_list, flatten

from datetime import datetime

def process_file(input_file = "data/arXiv-2106.04624v1/main.tex",  output_path="output/"):

    # params
    test_run = True
    text_volume_max  = 600 


    output_file = output_path + datetime.now().strftime(r"%y.%m.%d-%H")

    spark = get_spark_session("TTS CPU Inference")
    sc = spark.sparkContext()
    

    # Read, parse and convert LaTeX content
    rdd = sc.wholeTextFiles(input_file)

    processed_rdd = rdd.mapValues(preprocess_text)
    
    # Convert RDD to DataFrame
    df_processed = processed_rdd.map(lambda x: Row(filename=x[0], 
                                               text=x[1][0], 
                                               tables=x[1][1], 
                                               figures=x[1][2])).toDF()

    #FORK: only text continues forward. tables and figures to be implemented in different pipeline

    # split converted text into chunks
    df_chunks = df_processed.withColumn("chunks", udf_split_text(df_processed["text"])) \
        .selectExpr("filename", "explode(chunks) as sentence")
  

    chunks = df_chunks.withColumn("index", monotonically_increasing_id()) \
        .selectExpr("*"," length(sentence) as text_len")
    
    # for test runs I want to process just 15 chunks
    if test_run:
        chunks = chunks.offset(295).limit(15)


    # Partition by cumulative text volume
    text_volume_window = (Window.orderBy(desc('text_len'))
                .rowsBetween(Window.unboundedPreceding, 0))
    # TODO: maybe can use partitionng here for separating whole text into chapters?     

    step1 = chunks.withColumn('cum_text_volume', F.sum('text_len').over(text_volume_window)) \
        .withColumn('part', floor(col('cum_text_volume')/lit(text_volume_max)) ) 

    nparts =  step1.select((lit(1) + F.max("part")).alias("npart")).first()
    # this is bad. need to find way to bypass this pipeline leak

    # perform the TTS and vocoder inference
    processed = step1.repartitionByRange(nparts[0], "part") \
        .withColumn("prediction", predict_batch_udf(col("sentence"))).cache()\
            .select("index", "sentence", "text_len", "prediction.*") \
                .sort("index")

    # combine into single waveform
    wf = processed.agg(flatten(collect_list(col("waveform")))).alias("speech")

    output_sound(wf,output_file)




def gipis_process_batch(input_path, output_path):
    """Batch mode processing"""
    spark = get_spark_session("BatchWaveformProcessing")
    df = spark.read.parquet(input_path)

    df = compute_waveform_lengths(df)
    result_df = concatenate_waveforms(df, "order_column")

    result_df.write.mode("overwrite").parquet(output_path)
    spark.stop()

def process_stream(kafka_topic, kafka_servers, output_type, output_path=None):
    """Streaming mode processing"""
    spark = get_spark_session("StreamingWaveformProcessing", streaming=True)

    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "latest") \
        .load()

    # Extract value and cast to string (assuming JSON input)
    df = df.selectExpr("CAST(value AS STRING)")

    # Transform data
    df = compute_waveform_lengths(df)

    # Define output sink
    query = None
    if output_type == "kafka":
        query = df.selectExpr("CAST(value AS STRING) AS key", "value") \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_servers) \
            .option("topic", "processed_waveforms") \
            .option("checkpointLocation", "/tmp/kafka_checkpoint/") \
            .start()
    
    elif output_type == "hdfs" or output_type == "fs":
        query = df.writeStream \
            .format("parquet") \
            .option("path", output_path) \
            .option("checkpointLocation", "/tmp/parquet_checkpoint/") \
            .start()

    elif output_type == "spark_pipeline":
        query = df.writeStream \
            .format("memory") \
            .queryName("waveform_table") \
            .start()

    # Wait for termination
    query.awaitTermination()