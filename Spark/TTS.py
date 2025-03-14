# 5. Entry Point (main.py)

# This allows switching between batch and streaming dynamically.
import os
import sys
sys.path.append(os.getcwd())

import argparse
import threading
from pipeline import process_file, process_stream
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    
    logger.info("Starting pipeline...")

    parser = argparse.ArgumentParser(description="Waveform Processing Pipeline")
    parser.add_argument("--mode", choices=["batch", "stream"], required=True, help="Run in batch or streaming mode")
    parser.add_argument("--input", help="Path to input data (batch mode)")
    parser.add_argument("--output", help="Path to output data (batch mode)")
    parser.add_argument("--kafka-topic", help="Kafka topic (streaming mode)")
    parser.add_argument("--kafka-servers", help="Kafka bootstrap servers (streaming mode)")
    parser.add_argument("--output-type", choices=["kafka", "hdfs", "fs", "spark_pipeline"], help="Output type for streaming")
    
    args = parser.parse_args()

    if args.mode == "batch":
        if not args.input or not args.output:
            print("Error: Batch mode requires --input and --output")
            return
        process_file(args.input, args.output)

    elif args.mode == "stream":
        if not args.kafka_topic or not args.kafka_servers or not args.output_type:
            print("Error: Streaming mode requires --kafka-topic, --kafka-servers, and --output-type")
            return
        
        # Run streaming in a separate thread so it can be stopped gracefully
        stream_thread = threading.Thread(target=process_stream, args=(args.kafka_topic, args.kafka_servers, args.output_type, args.output))
        stream_thread.start()
        try:
            stream_thread.join()
        except KeyboardInterrupt:
            print("Shutting down gracefully...")

if __name__ == "__main__":
    main()