# 4. Pipeline Logic (pipeline.py)

# This handles both batch and streaming.

from config import get_spark_session
import config as conf
from processing import compute_waveform_lengths, concatenate_waveforms, cum_text_volume, output_sound, save_to_disk, custom_write_function
from processing import preprocess_text_udf, split_text_into_chunks_udf, predict_batch_udf , sentences_schema, concat_waveforms, write_request, write_row

from pyspark.sql import functions as F, Row
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import pandas_udf, PandasUDFType, udf, col, lit, desc, floor, monotonically_increasing_id
from pyspark.sql.functions import collect_list, flatten, current_timestamp, date_format, expr, window, any_value
from pyspark.sql.functions import posexplode, explode, input_file_name, col, concat_ws

from datetime import datetime

def batch_empty(input_file = "data/arXiv-2106.04624v1/main.tex", output_path="output/"):
    spark = get_spark_session(conf.app_name)  

    from processing import processed_schema

    empty_df = spark.createDataFrame([], processed_schema) # spark is the Spark Session
    
    empty_df.show()
    df_wf = empty_df.transform(tts_core)

    df_wf.show()

    df_wf.foreach(write_row)


def process_file(input_file = "data/arXiv-2106.04624v1/main.tex", output_path="output/"):
 
    spark = get_spark_session(conf.app_name)    
    
    # Read text files into DataFrame with columns "filename", "request_id" and "content"
    # input_file can also be a directory of files
    # add request_id (a time stamp) for tracking jobs in cluster
    df_requests = spark.read.text(input_file).withColumn("request_id", F.element_at(F.split(F.input_file_name(), '/'),-2))\
                    .withColumn("timestamp",current_timestamp())\
                    .groupBy("timestamp","request_id") \
                    .agg(concat_ws("\n", collect_list("value")).alias("content"))\
                    .selectExpr("*", "struct(timestamp as start, timestamp as end) as window") \

    df_requests.show()                     
    # preprocess LaTeX text 
    df_processed = df_requests.withColumn("processed", preprocess_text_udf(col("content")))

    # Extract text, tables, and figures into separate columns
    df_processed = df_processed.select(
        "timestamp", "window","request_id",
        df_processed["processed.text"].alias("text"),
        df_processed["processed.tables"].alias("tables"),
        df_processed["processed.figures"].alias("figures")
    )
    
    #run core pipeline
    df_wf = tts_core(df_processed)
    
    # save processed speech to disk
    df_wf.foreach(write_row)
def tts_core(df_texts):   
 
    # Apply UDF (returns an array of structs)
    df_chunks = df_texts.withColumn("chunks", split_text_into_chunks_udf(col("text")))

    # Explode chunks into multiple rows
    df_chunks = df_chunks.select(
        "timestamp","request_id",
        posexplode("chunks").alias("index", "sentence")   # This creates multiple rows per file
    )
    # #FORK: only text continues forward. tables and figures to be implemented in different pipeline

    chunks = df_chunks\
        .selectExpr("timestamp","request_id", "index", " sentence ",  " length(sentence) as text_len")
    
    chunks.groupby(col("timestamp"), col("request_id"))\
          .agg(F.count("sentence").alias("sentences"))\
          .select("timestamp","request_id","sentences")\
          .show()

    # for test runs I want to process just 5 chunks per request
    if conf.test_run:
        chunks = chunks.orderBy("index", "request_id").offset(100).limit(8)

    chunks.show()

    # Partition by cumulative text volume
    text_volume_window = (Window.orderBy(desc('text_len'))
                .rowsBetween(Window.unboundedPreceding, 0))
    # TODO: maybe can use partitionng here for separating whole text into chapters?     

    step1 = chunks.withColumn('cum_text_volume', F.sum('text_len').over(text_volume_window)) \
        .withColumn('part', floor(col('cum_text_volume')/lit(conf.text_volume_max)) ) 

    #nparts =  step1.select((lit(1) + F.max("part")).alias("npart")).first()
    # this is bad. need to find way to bypass this pipeline leak

    # perform the TTS and vocoder inference
    processed = step1.repartition("part") \
        .withColumn("prediction", predict_batch_udf(col("sentence"))).cache()\
            .select("timestamp","request_id", "index", "sentence", "text_len", "prediction.*") \
                .sort("index")

    # combine into single waveform
    wf = processed.groupBy("timestamp","request_id")\
        .agg(flatten(collect_list(col("waveform"))).alias("speech"))\
            
    # TODO: maybe we can (should?) recombine this with df_processed? 
    return wf

def stream_batch(batchDF, batchId):

    batchDF.show()
    df_wf = tts_core(batchDF)
    print("batch id: ", batchId)
    df_wf.show()
    output_schema = batchDF.schema    
    df_wf.foreach(write_row)
    #df_wf.groupBy("request_id").applyInPandas(write_request, schema=output_schema)


def process_stream(kafka_topic, kafka_servers, output_type, output_path=None):
    print("Streaming mode processing turned on")
    spark = get_spark_session(conf.app_name, streaming=True)

    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()

    # Extract value and cast to string (assuming JSON input)
    df_requests = df.selectExpr("CAST(key AS STRING) as request_id",
                                 "CAST(value AS STRING) as content",
                                 "timestamp as timestamp") \

    # preprocess LaTeX text 
    df_processed = df_requests.withColumn("processed", preprocess_text_udf(col("content")))

    # Extract text, tables, and figures into separate columns
    df_processed = df_processed.select(
        "timestamp","request_id",
        df_processed["processed.text"].alias("text"),
        df_processed["processed.tables"].alias("tables"),
        df_processed["processed.figures"].alias("figures")
    )
    import time
    job_time = f"{int(time.time())}"  
    #run core pipeline
    #df_wf = df_processed.transform(batch_tts)
    #df_wf = pipeline_core(df_requests)
    # Define output sink

    query = df_processed \
            .writeStream \
            .queryName(f"{output_type}_{job_time}") \
            .foreachBatch(stream_batch) \
            .outputMode("append") \
            .option("checkpointLocation", "/tmp/Spark/checkpoints/" + output_type + "/" + job_time) \
            .trigger(processingTime="5 seconds")\
            .start()
    

    query.awaitTermination()
    print("Spark streaming turned off")


def stream_outputs():
  
    if output_type == "kafka":
        pass
        # query = df.selectExpr("CAST(value AS STRING) AS key", "value") \
        #     .writeStream \
        #     .format("kafka") \
        #     .option("kafka.bootstrap.servers", kafka_servers) \
        #     .option("topic", "processed_waveforms") \
        #     .option("checkpointLocation", "/tmp/kafka_checkpoint/") \
        #     .start()
 
    # save processed speech to disk
    # .outputMode("complete").trigger(processingTime="10 seconds") \
    elif output_type == "hdfs" or output_type == "fs":
        query = df_wf \
            .writeStream \
            .queryName(f"{output_type}_{job_time}") \
            .foreachBatch(custom_write_function) \
            .outputMode("append") \
            .option("checkpointLocation", "/tmp/Spark/checkpoints/" + output_type + "/" + job_time) \
            .start()
        
    elif output_type == "parquet":
        query = df_wf.writeStream \
            .queryName(f"{output_type}_{job_time}") \
            .outputMode("append") \
            .format("parquet") \
            .option("path", conf.output_path ) \
            .option("checkpointLocation", "/tmp/Spark/checkpoints/" + output_type + "/" + job_time) \
            .start()
             # Force re-polling Kafka every 10s               .trigger(processingTime="10 seconds") \

    elif output_type == "spark_pipeline":
        pass

    elif output_type == "console":
        query = df_wf.writeStream\
            .format("console")\
            .outputMode("append") \
            .option("truncate", False) \
            .start()
        
        # query = df.writeStream \
        #     .format("memory") \
        #     .queryName("waveform_table") \
        #     .start()
    query.explain()
    # while query.isActive:
    #     print(query.lastProgress)
    #     time.sleep(10)
    # Wait for termination
    
    spark.streams.awaitAnyTermination()
    #query.awaitTermination()
    print("Spark streaming turned off")