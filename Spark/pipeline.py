# Pipeline Logic (pipeline.py)

# This handles both batch and streaming.

from config import get_spark_session
import config as conf
from processing import preprocess_text_udf, split_text_into_chunks_udf, predict_batch_udf, write_row

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, lit, desc, floor
from pyspark.sql.functions import collect_list, flatten, current_timestamp
from pyspark.sql.functions import posexplode, col, concat_ws, spark_partition_id


def batch_empty(input_file = "data/arXiv-2106.04624v1/main.tex", output_path="output/"):
    spark = get_spark_session(conf.app_name)  

    from processing import processed_schema

    empty_df = spark.createDataFrame([], processed_schema) 
    
    empty_df.show()
    df_wf = empty_df.transform(tts_core)

    df_wf.show()

    df_wf.foreach(write_row)


def process_batch(input_file = "data/arXiv-2106.04624v1/main.tex", output_path="output/"):
 
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
    #FORK: only text continues forward. tables and figures to be implemented in different pipeline
    
    #run core pipeline
    df_wf = tts_core(df_processed)
    
    # save processed speech to disk
    df_wf.foreach(write_row)


def tts_core(df_texts):   
 
    # Apply UDF (returns an array of structs)
    df_chunks = df_texts.withColumn("chunks", split_text_into_chunks_udf(col("text")))

    # Explode chunks into multiple rows
    df_chunks = df_chunks.select(
        "request_id",
        posexplode("chunks").alias("index", "sentence")   # This creates multiple rows per file
    )

    chunks = df_chunks\
        .selectExpr("request_id", "index", " sentence ",  " length(sentence) as text_len")
    
    chunks.groupby(col("request_id"))\
          .agg(F.count("sentence").alias("sentences"))\
          .select("request_id","sentences")\
          .show()

    # for test runs I want to process just 5 chunks per request
    if conf.test_run:
        chunks = chunks.orderBy("index", "request_id").offset(120).limit(conf.test_size)

    # Partition by cumulative text volume
    text_volume_window = (Window.orderBy(desc('text_len'))
                .rowsBetween(Window.unboundedPreceding, 0))
    # TODO: maybe can use partitionng here for separating whole text into chapters?     

    step1 = chunks.withColumn('cum_text_volume', F.sum('text_len').over(text_volume_window)) \
        .withColumn('part', floor(col('cum_text_volume')/lit(conf.text_volume_max))) 
  
    nparts =  step1.select((lit(1) + F.max("part")).alias("npart")).first()
    # this is bad. need to find way to bypass this pipeline leak
    np = nparts[0] if nparts[0] else 1

    # perform the TTS and vocoder inference
    repartitioned = step1.repartition(np, col("part")) #,10, 
    

    step1.withColumn("partitionId", spark_partition_id())\
         .show(110)

    print(f"Number of partitions: {repartitioned.rdd.getNumPartitions()}")
    
    processed = repartitioned\
        .withColumn("prediction", predict_batch_udf(col("sentence")))\
        .select("*", "prediction.*")\
        .drop("prediction") 
    
    processed.withColumn("processed",col("seq_len")).show()

    repartagain = processed.repartition(2,col("request_id")) 

    df_wf = repartagain\
                .groupby("request_id")\
                .agg(F.sort_array(
                     F.collect_list(
                          F.struct("request_id","index", "waveform")))
                    .alias("wfs"))\
                .select("request_id",
                        flatten("wfs.waveform").alias("speech"),
                        concat_ws(",",col("wfs.index")).alias("order"))                        

    df_wf.show()
    # TODO: maybe we can (should?) recombine this with df_processed? 
    return df_wf, processed

def stream_batch(batchDF, batchId):

    batchDF.show()
    df_wf, processed = tts_core(batchDF)
    print("batch id: ", batchId)
    #df_wf.show()
    stream_outputs(df_wf)
    if conf.test_run:
        processed.drop("waveform")\
             .write\
             .mode('append')\
             .csv(conf.output_path + "/intermediate/csv/")
        
        df_wf.drop("speech")\
             .write\
             .mode('append')\
             .csv(conf.output_path + "/df_wf/csv/")


def process_stream(kafka_topic, kafka_servers, output_types, output_path=None):
    print("Streaming mode processing turned on")

    conf.output_types = output_types
    conf.output_path = output_path

    spark = get_spark_session(conf.app_name, streaming=True)

    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "latest") \
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
    #FORK: only text continues forward. tables and figures to be implemented in different pipeline

    # Define output sink
    query = df_processed \
            .writeStream \
            .foreachBatch(stream_batch) \
            .outputMode("append") \
            .option("checkpointLocation", "/tmp/Spark/checkpoints/")\
            .trigger(processingTime="5 seconds")\
            .start()

    query.awaitTermination()
    print("Spark streaming turned off")


def stream_outputs(df_wf):

    if "kafka" in conf.output_types:
        pass
 
    # save processed speech to disk
    if "hdfs" in conf.output_types or "fs" in conf.output_types:
        df_wf.foreach(write_row)

    if "parquet" in conf.output_types:
        df_wf.write\
             .mode('append')\
             .partitionBy("request_id")\
             .parquet(conf.output_path + "/parquet/")

    if "spark_pipeline" in conf.output_types:
        pass

    if "console" in conf.output_types:
        pass
    # while query.isActive:
    #     print(query.lastProgress)
    #     time.sleep(10)  