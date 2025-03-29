# 4. Pipeline Logic (pipeline.py)

# This handles both batch and streaming.

from config import get_spark_session
import config as conf
from processing import compute_waveform_lengths, concatenate_waveforms, cum_text_volume, output_sound, save_to_disk, custom_write_function
from processing import preprocess_text_udf, split_text_into_chunks_udf, predict_batch_udf , sentences_schema, concat_waveforms, write_request, write_row

from pyspark.sql import functions as F, Row
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import pandas_udf, PandasUDFType, udf, col, lit, desc, floor, monotonically_increasing_id
from pyspark.sql.functions import collect_list, flatten, current_timestamp, date_format, expr, window, any_value, first_value, last_value
from pyspark.sql.functions import posexplode, explode, input_file_name, col, concat_ws

from datetime import datetime

def batch_empty(input_file = "data/arXiv-2106.04624v1/main.tex", output_path="output/"):
    spark = get_spark_session(conf.app_name)  

    from processing import processed_schema

    empty_df = spark.createDataFrame([], processed_schema) # spark is the Spark Session
    
    empty_df.show()
    df_wf = empty_df.transform(batch_tts)

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
    df_wf = batch_tts(df_processed)

    # 7. Join the assembled speech back to the original DataFrame.
    #df_wf = df_processed.join(transformed.select("window","request_id", "speech"), on=["window","request_id"], how="left")
    # save processed speech to disk
    df_wf.foreach(write_row)
   

def batch_tts(microBatchDF):
    # if microBatchDF.rdd.isEmpty():
    #     return microBatchDF.withColumn("sorted_waveforms", "select window")

        # Apply UDF (returns an array of structs)
    df_chunks = microBatchDF.select("timestamp", "request_id", "text")\
                            .withColumn("chunks", split_text_into_chunks_udf(col("text")))

    # Explode chunks into multiple rows
    df_exploded = df_chunks.select("timestamp", "request_id", posexplode("chunks").alias("index", "sentence")) 
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
    chunks = df_exploded.groupBy("timestamp", "request_id").applyInPandas(cum_text_volume, sentences_schema)



    
    # 4. Repartition by request_id and part to distribute the workload.
    repartitioned = chunks.withColumn('part', floor(col('cum_text_volume')/lit(conf.text_volume_max))) \
                          .repartition("timestamp","request_id", "part")
    

    # options: 
    # 1. somehow run the whole chunks processing as a spark subjob on every row of df_processed
    # 2. find another way to calcualte cumulative sum inside structured streaming, without window 

    # perform the TTS and vocoder inference
    processed = repartitioned\
        .withColumn("prediction", predict_batch_udf(col("sentence"))) \
        .select("*", "prediction.*")\
        

    assembled = processed#.withColumn("ts",  expr("window.start")) \
                         #.withWatermark("ts", "10 seconds")\

                      
                        
    #window("ts", "5 seconds")

    collected = assembled.withWatermark("timestamp", "10 minutes") \
                         .groupBy(window(col("timestamp"), "5 seconds"), col("request_id"))\
                         .agg(F.sort_array(F.collect_list(F.struct("index", "waveform")), asc=True).alias("sorted_waveforms"))
    #F.sort_array(F.struct("index", "waveform"), asc=True)   #, col("request_id"), col("timestamp")

    collected.select("window", "request_id", F.slice("sorted_waveforms",1,5))\
                .writeStream\
                .format("console")\
                .option("truncate", True)\
                .outputMode("complete") \
                .start()  

    # microBatchDF.writeStream\
    #             .format("console")\
    #             .option("truncate", True)\
    #             .outputMode("complete") \
    #             .start()  
    # 7. Join the assembled speech back to the original DataFrame.
    df_wf = collected.select("window", "request_id", F.flatten("sorted_waveforms.waveform").alias("speech"))
    # df_wf.writeStream\
    #             .format("console")\
    #             .option("truncate", True)\
    #             .outputMode("append") \
    #             .start()  

    df_wf.select("window", "request_id", F.slice("speech",1,6))\
            .writeStream\
            .format("console")\
            .option("truncate", True)\
            .outputMode("complete") \
            .start()  

    return df_wf 

    
    #.withColumn("speech", concat_waveforms(F.col("sorted_waveforms"))).drop("sorted_waveforms")
    #return processed   




        # .withColumn("mel_lengths", "prediction.mel_lengths") \
        # .withColumn("seq_len", "prediction.seq_len")

            # .select("timestamp","request_id", "index", "sentence", "text_len", "prediction.*") 
               
    #processed.show()

    # 6. Reassemble: Group by request and sort by index to concatenate the waveform.    
 #.orderBy("index")
    # .agg(
    #     F.sort_array(F.collect_list(F.struct("index", "waveform")), asc=True).alias("sorted_waveforms")
    # )

    #collected.show()
    #result.show()

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
        .option("failOnDataLoss", "true") \
        .load() \
        
# .option("groupId", "fixed_consumer_group")\
    # Extract value and cast to string (assuming JSON input)
    df_requests = df.selectExpr("CAST(key AS STRING) as request_id",
                                "CAST(value AS STRING) as content",
                                "timestamp as timestamp") \

        # .groupBy(window(col("timestamp"), "5 seconds"), col("request_id")) \
        # .agg(any_value("cont", True).alias("content"), F.count("*").alias("batch_size"))\
        

    #.filter(F.col("content").isNotNull())  # Ignore null rows
    #F.min("timestamp").alias("timestamp"), 

    # df_requests.writeStream\
    #             .format("console")\
    #             .option("truncate", True)\
    #             .outputMode("append") \
    #             .start()
    
    # preprocess LaTeX text 
    df_processed = df_requests.withColumn("processed", preprocess_text_udf(col("content")))

    # Extract text, tables, and figures into separate columns
    df_processed = df_processed.select(
        col("timestamp"),col("request_id"),
        df_processed["processed.text"].alias("text"),
        df_processed["processed.tables"].alias("tables"),
        df_processed["processed.figures"].alias("figures")
    )

    #run core pipeline
    transformed = df_processed.transform(batch_tts)

    # #transformed = batch_tts(df_processed)
    # df_processed.select(col("timestamp"), col("request_id"))\
    #         .writeStream\
    #         .format("console")\
    #         .option("truncate", False)\
    #         .outputMode("append") \
    #         .start()
    #         #.trigger(availableNow=True) \
    
    transformed.select("window", "request_id", F.expr("slice(speech, 1, 6)"))\
        .writeStream\
        .format("console")\
        .option("truncate", False)\
        .outputMode("append") \
        .start()

    df_wf = df_processed.join(transformed, 
                              [df_processed["request_id"] == transformed["request_id"], 
                               df_processed["timestamp"] >= transformed["window.start"],
                            #    df_processed["timestamp"] <= transformed["window.end"]
                             ],
                              how="Left")\
                              .select(df_processed["*"], transformed["speech"])
    

# Stream-stream LeftOuter join between two streaming DataFrame/Datasets is not supported
#  without a 
# watermark in the join keys, 
# or a watermark on the nullable side 
# and an appropriate range condition;

    df_wf.withColumn("speech", F.expr("slice(speech, 1, 7)"))\
        .writeStream\
        .format("console")\
        .option("truncate", True)\
        .outputMode("append") \
        .start()

#     df_wf = df_processed.join(transformed, 
#                               [df_processed.timestamp == transformed.window.start,
#                                df_processed.request_id == transformed.request_id], 
#                                how="Inner")
#   expr("""
#     clickAdId = impressionAdId AND
#     clickTime >= transformed.window.start AND
#     clickTime <= impressionTime + interval 1 hour
#     """),


    #df_wf = transformed

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