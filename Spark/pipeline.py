# 4. Pipeline Logic (pipeline.py)

# This handles both batch and streaming.

from config import get_spark_session
from processing import compute_waveform_lengths, concatenate_waveforms, output_sound, save_to_disk
from processing import preprocess_text_udf, split_text_into_chunks_udf, predict_batch_udf 
from utils import zip_project

from pyspark.sql import functions as F, Row
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import pandas_udf, PandasUDFType, udf, col, lit, desc, floor, monotonically_increasing_id
from pyspark.sql.functions import collect_list, flatten, current_timestamp, date_format
from pyspark.sql.functions import explode
from datetime import datetime

def process_file(input_file = "data/arXiv-2106.04624v1/main.tex", output_path="output/"):

    # params
    test_run = True
    text_volume_max  = 600 
    
    #logger.info("Zipping project for distribution to Spark workers")
    zip_project(
        project_dir="ArticleReader",
        zip_filename="ArticleReader.zip",
        exclude_patterns=["trash/*"])
    
    zip_project(
        project_dir="Spark",
        zip_filename="Spark.zip",
        exclude_patterns=["trash/*"])

    #output_file = output_path + datetime.now().strftime(r"%y.%m.%d-%H")

    spark = get_spark_session("TTS CPU Inference")
    sc = spark.sparkContext
    sc.addPyFile("ArticleReader.zip")
    sc.addPyFile("Spark.zip")

    from pyspark.sql.functions import input_file_name, col, concat_ws

    # Read text files into DataFrame with columns "filename", "request_id" and "content"
    # input_file can also be a directory of files
    df_whole = spark.read.text(input_file).withColumn("filename", input_file_name()) \
        .withColumn("request_id",date_format(current_timestamp(),"yy.MM.dd-HH.mm.ss.SSS")\
                    .alias("request_id")) \
        .groupBy("filename","request_id") \
        .agg(concat_ws("\n", collect_list("value")).alias("content"))
    
    # # Read text files and aggregate content per file
    # df_whole = spark.read.text("path/to/files") \
    #     .withColumn("filename", input_file_name()) \

    df_processed = df_whole.withColumn("processed", preprocess_text_udf(col("content")))

    # Extract text, tables, and figures into separate columns
    df_processed = df_processed.select(
        "filename","request_id",
        df_processed["processed.text"].alias("text"),
        df_processed["processed.tables"].alias("tables"),
        df_processed["processed.figures"].alias("figures")
    )

    # Apply UDF (returns an array of structs)
    df_chunks = df_processed.withColumn("chunks", split_text_into_chunks_udf(col("text")))

    # Explode chunks into multiple rows
    df_chunks = df_chunks.select(
        "filename","request_id",
        explode("chunks").alias("sentence")  # This creates multiple rows per file
    )
    # #FORK: only text continues forward. tables and figures to be implemented in different pipeline

    chunks = df_chunks.withColumn("index", monotonically_increasing_id()) \
        .selectExpr("filename","request_id", "index", " sentence ",  " length(sentence) as text_len")
    
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
            .select("filename","request_id", "index", "sentence", "text_len", "prediction.*") \
                .sort("index")

    # combine into single waveform
    wf = processed.groupBy("filename","request_id")\
        .agg(flatten(collect_list(col("waveform"))).alias("speech"))\
            
    # TODO: maybe we can (should?) recombine this with df_processed? 

    # save processed speech to disk
    wf.foreach(save_to_disk)




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