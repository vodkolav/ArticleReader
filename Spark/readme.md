

6. Running the Pipeline
Batch Mode

python main.py --mode batch --input data/arXiv-2106.04624v1/main.tex --output output

Streaming Mode

python main.py --mode stream --kafka-topic raw_waveforms --kafka-servers localhost:9092 --output-type kafka

or saving to HDFS:

python main.py --mode stream --kafka-topic raw_waveforms --kafka-servers localhost:9092 --output-type hdfs --output hdfs://namenode:9000/processed_waveforms

7. Stopping the Streaming Pipeline

Since awaitTermination() blocks execution, you can stop the service by pressing CTRL+C or sending a termination signal.

To automate stopping:

pkill -f "python main.py --mode stream"