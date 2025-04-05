# Spark & Kafka Configurations (config.py)

# Define a function to create a Spark session that supports both batch and streaming.
import os
from pyspark.sql import SparkSession
import torch
import logging
logger = logging.getLogger(__name__)
from utils import zip_project

# params
app_name="TTS CPU Inference"
test_run = True # False # 
test_size = 20
text_volume_max = 330  # will need to be tuned for specific cluster machines
chunk_size = 80
output_path="output/"
output_types = ["fs","parquet"]
articles_topic="articles"
# simulating a cluster with 2 workers
workers = 1 # 2 #
cpus_limit =  int(os.cpu_count()/ workers) -1 
mem_limit = "14g" # prod: "16g"/ workers

def get_spark_session(app_name="TTS CPU Inference", streaming=False):
    # Configure PyTorch for CPU parallelism
    torch.set_num_threads(cpus_limit)

    builder = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.executor.cores", cpus_limit) \
        .config("spark.executor.instances", workers) \
        .config("spark.executor.memory", mem_limit) \
        .config("spark.task.cpus", cpus_limit) \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
        .config("spark.sql.execution.arrow.pyspark.enabled","true") \
        .config("log4j.logger.org.apache.spark","DEBUG")\
        .config("log4j.logger.org.apache.kafka","DEBUG") \
        .config("spark.sql.streaming.metricsEnabled", "true")\
        .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
        
        # Additional configs that might be useful in future:
        # .config("spark.dynamicAllocation.enabled", "false") \
        #         .config("spark.sql.shuffle.partitions", 20) \
        # .config("spark.executor.resource.gpu.amount", "1") \
        # .config("spark.executor.memoryOverhead", "<memory>"
    if streaming:
        builder = builder.config("spark.sql.streaming.schemaInference", "true")
    spark = builder.getOrCreate()
    #spark.sparkContext.setLogLevel("DEBUG")
    distribute_project_code(spark)
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Using {cpus_limit} CPU cores for inference.")   

    return spark

def distribute_project_code(spark):
    # distributes user code to all workers
    logger.info("Zipping project for distribution to Spark workers")
    zip_project(
        project_dir="ArticleReader",
        zip_filename="ArticleReader.zip",
        exclude_patterns=["trash/*"])
    
    zip_project(
        project_dir="Spark",
        zip_filename="Spark.zip",
        exclude_patterns=["trash/*"])

    sc = spark.sparkContext
    sc.addPyFile("ArticleReader.zip")
    sc.addPyFile("Spark.zip")