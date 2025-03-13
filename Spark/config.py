# 2. Spark & Kafka Configurations (config.py)

# Define a function to create a Spark session that supports both batch and streaming.
import os
from pyspark.sql import SparkSession
import torch
import logging
logger = logging.getLogger(__name__)

app_name="TTS CPU Inference"

def get_spark_session(app_name="TTS CPU Inference", streaming=False):

    # simulating a cluster with 2 workers
    workers = 2
    cpus_limit =  int(os.cpu_count()/ workers) -1 
    mem_limit = "2g" # prod: "16g"/ workers

    # Configure PyTorch for CPU parallelism
    torch.set_num_threads(cpus_limit)

    builder = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.executor.cores", cpus_limit) \
        .config("spark.executor.instances", workers) \
        .config("spark.executor.memory", mem_limit) \
        .config("spark.task.cpus", cpus_limit) \
        .config("spark.dynamicAllocation.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "1") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        # Additional configs that might be useful in future:
        # .config("spark.executor.resource.gpu.amount", "1") \
        # .config("spark.executor.memoryOverhead", "<memory>"
    if streaming:
        builder = builder.config("spark.sql.streaming.schemaInference", "true")
    spark = builder.getOrCreate()

    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Using {cpus_limit} CPU cores for inference.")   

    return spark

