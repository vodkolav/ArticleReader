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
    df_wf = batch_tts(df_processed)
    
    # save processed speech to disk
    df_wf.foreach(write_row)
   

def batch_tts(microBatchDF):
    # if microBatchDF.rdd.isEmpty():
    #     return

        # Apply UDF (returns an array of structs)
    df_chunks = microBatchDF.select("window", "request_id", "text")\
                            .withColumn("chunks", split_text_into_chunks_udf(col("text")))

    # Explode chunks into multiple rows
    df_exploded = df_chunks.select("window", "request_id", posexplode("chunks").alias("index", "sentence")) 
                          # .select("*","chunk.*")    
    # #FORK: only text continues forward. tables and figures to be implemented in different pipeline

    df_exploded = df_exploded.withColumn("text_len", F.length(F.col("sentence")))
    
    # for test runs I want to process just 15 chunks
    if conf.test_run:
        df_exploded = df_exploded.filter(col("index").between(lit(301),lit(305)))

    # chunks = chunks.withColumn("timestamp",col("window.start"))\
    #                .withWatermark("timestamp", "1 minute") \
    #step1 = chunks.groupBy("request_id").applyInPandas(cum_text_volume, schema=sentences_schema)


    # You now have a streaming DataFrame with an additional 'part' column,
    # so that all rows with the same 'part' within a request can be treated as one batch.
    # You can then repartition using this column if desired:
    #result = result.repartition("part")
    

    # Partition by cumulative text volume .partitionBy("window", "request_id" )
    # text_volume_window = Window.partitionBy("window", "request_id") \
    #                     .orderBy("timestamp", desc("text_len")) \
    #                     .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        #`withColumn('cum_text_volume', F.sum('text_len').over(text_volume_window)) \
     # Apply the UDF
    # chunks = chunks.withColumn("cum_text_volume", 
    #                            cum_text_volume(chunks.select("window", "request_id", "text_len")))
                  


    # Apply the function using applyInPandas
    chunks = df_exploded.groupBy("window", "request_id").applyInPandas(cum_text_volume, sentences_schema)



    
    # 4. Repartition by request_id and part to distribute the workload.
    repartitioned = chunks.withColumn('part', floor(col('cum_text_volume')/lit(conf.text_volume_max))) \
                          .repartition("window","request_id", "part")
    

    # options: 
    # 1. somehow run the whole chunks processing as a spark subjob on every row of df_processed
    # 2. find another way to calcualte cumulative sum inside structured streaming, without window 

    # perform the TTS and vocoder inference
    processed = repartitioned\
        .withColumn("prediction", predict_batch_udf(col("sentence"))) \
        .select("*", "prediction.*")
        # .withColumn("mel_lengths", "prediction.mel_lengths") \
        # .withColumn("seq_len", "prediction.seq_len")

            # .select("timestamp","request_id", "index", "sentence", "text_len", "prediction.*") 
               
    #processed.show()

    # 6. Reassemble: Group by request and sort by index to concatenate the waveform.
    assembled = processed.groupBy("window","request_id")
                         
    #.orderBy("index")
    
    # .agg(
    #     F.sort_array(F.collect_list(F.struct("index", "waveform")), asc=True).alias("sorted_waveforms")
    # )
    
    collected = assembled.agg(F.sort_array(F.collect_list(F.struct("index", "waveform")), asc=True).alias("sorted_waveforms"))
    collected = collected.withColumn("speech", F.flatten("sorted_waveforms.waveform"))
    
    #.withColumn("speech", concat_waveforms(F.col("sorted_waveforms"))).drop("sorted_waveforms")

    #collected.show()
  
    # 7. Join the assembled speech back to the original DataFrame.
    result = microBatchDF.join(collected.select("window","request_id", "speech"), on=["window","request_id"], how="left")
    #result.show()
    return result

    # wf = processed.withWatermark("timestamp", "1 minute") \
    #     .groupBy("timestamp", "request_id") \
    #     .agg(expr("sort_array(collect_list(struct(index, waveform))) as sorted_waveforms"))

    # wf = wf.withColumn("speech", expr("flatten(transform(sorted_waveforms, x -> x.waveform))"))


    # combine into single waveform
    # wf = processed.withWatermark("timestamp", "1 minute") \
    #     .sort("index") \
    #         .groupBy("timestamp","request_id") \
    #             .agg(flatten(collect_list(col("waveform"))).alias("speech"))\
            
    # TODO: maybe we can (should?) recombine this with df_processed? 
    #return wf

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
                                 "CAST(value AS STRING) as cont",
                                 "timestamp as timestamp") \
        .withWatermark("timestamp", "1 minute") \
        .groupBy(window(col("timestamp"), "1 minute"), "request_id") \
        .agg(any_value("cont").alias("content"))
    #F.min("timestamp").alias("timestamp"), 

    # preprocess LaTeX text 
    df_processed = df_requests.withColumn("processed", preprocess_text_udf(col("content")))

    # Extract text, tables, and figures into separate columns
    df_processed = df_processed.select(
        "window","request_id",
        df_processed["processed.text"].alias("text"),
        df_processed["processed.tables"].alias("tables"),
        df_processed["processed.figures"].alias("figures")
    )

    #run core pipeline
    df_wf = df_processed.transform(batch_tts)
    #df_wf = pipeline_core(df_requests)
    # Define output sink
    import time
    job_time = f"{int(time.time())}"      

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
        # query = df.writeStream \
        #     .format("memory") \
        #     .queryName("waveform_table") \
        #     .start()
    while query.isActive:
        print(query.lastProgress)
        time.sleep(10)
    # Wait for termination
    query.awaitTermination()
    print("Spark streaming turned off")