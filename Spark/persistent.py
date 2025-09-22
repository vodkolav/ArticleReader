from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql.types import ArrayType, StringType
import numpy as np
import os
from pyspark import SparkFiles

# Assume 'model_path_on_hdfs' is the HDFS path to your large TTS model
# e.g., "hdfs:///user/models/tts_model.bin"
# This path needs to be consistent with the filename used in SparkFiles.get()
MODEL_HDFS_PATH = "hdfs:///models/my_tts_model.bin"

def make_tts_model_predict_fn():
    # This function runs ONCE per Python worker (executor core)
    # and the loaded model is cached.
    model = None
    try:
        # Access the model from the local path where SparkFiles staged it
        # The filename here must match the base name of the file added via spark.sparkContext.addFile()
        local_model_path = SparkFiles.get(os.path.basename(MODEL_HDFS_PATH))
        print(f"Loading TTS model from: {local_model_path} on worker {os.getpid()}")
        # Replace with actual model loading logic (e.g., PyTorch, TensorFlow)
        # For demonstration, a dummy class
        class DummyTTSModel:
            def __init__(self, path):
                self.path = path
                # Simulate loading a large model
                import time
                time.sleep(5) # Simulate heavy loading
                print(f"Model {path} loaded successfully on worker {os.getpid()}")

            def predict(self, batch_of_texts: np.ndarray) -> np.ndarray:
                # Simulate TTS inference
                return np.array([f"Generated audio for: {text}" for text in batch_of_texts])

        model = DummyTTSModel(local_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    # This inner function will be called for each batch of data within the partition
    def predict_batch(texts: np.ndarray) -> np.ndarray:
        if model is None:
            raise RuntimeError("Model not loaded!")
        return model.predict(texts)

    return predict_batch

# Define the UDF outside the foreachBatch function
# The return_type should match the output of your TTS model (e.g., ArrayType(FloatType) for audio embeddings)
tts_predict_udf = predict_batch_udf(
    make_tts_model_predict_fn,
    return_type=ArrayType(StringType()), # Example: assuming string output for simplicity
    batch_size=32 # Optimal batch size for inference, typically smaller than Spark partition size [6, 7]
)

def process_batch(df: DataFrame, batch_id: int):
    if not df.isEmpty():
        print(f"Processing batch {batch_id}")
        # Apply the UDF for inference
        # Assuming input DataFrame has a 'text_input' column
        result_df = df.withColumn("generated_audio", tts_predict_udf(col("text_input")))
        result_df.show(truncate=False)
        # Further processing or writing to sink
        # For example, write to Delta Lake
        # result_df.write.format("delta").mode("append").save("/path/to/output_delta_table")
    else:
        print(f"Batch {batch_id} is empty.")

# Main Structured Streaming logic (example usage setup)
# spark = SparkSession.builder.appName("TTSSparkStreaming").getOrCreate()
# spark.sparkContext.addFile(MODEL_HDFS_PATH) # IMPORTANT: Add model file here on the driver

# Example streaming source (e.g., Kafka, Rate source for testing)
# stream_df = spark.readStream.format("rate").option("rowsPerSecond", 5).load() \
#    .withColumn("text_input", concat(lit("Hello from row "), col("value"))) # Dummy text input

# query = stream_df.writeStream \
#    .foreachBatch(process_batch) \
#    .outputMode("append") \
#    .option("checkpointLocation", "/tmp/checkpoint/tts_inference") \
#    .trigger(processingTime="10 seconds") \
#    .start()

# query.awaitTermination()