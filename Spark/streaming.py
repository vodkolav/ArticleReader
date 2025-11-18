import time
import signal
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from multiprocessing import Process, Event
import os, fnmatch
from datetime import datetime
import argparse
import config as conf

class KafkaProducer:
    def __init__(self, topic, source, config, interval_sec = 10):
        self.topic = topic
        self.source = source
        self.config = config
        self.interval_sec = interval_sec
        self.stop_event = Event()
        self.admin_client = AdminClient({'bootstrap.servers': self.config['bootstrap.servers']})


    def feed_file(self, file):
        with open(self.source, 'r') as file:
            #reader = csv.DictReader(file)
            batch  = ""
            for i, row in enumerate(file):
                if self.stop_event.is_set():
                    break
                batch += row    
                if(i%self.rows_per_invl)==0:
                    
                    # Send row as JSON string to Kafka topic
                    print(f"\rfeeding row {i} : {i+self.rows_per_invl-1} to kafka: " 
                            + batch.replace("\n", "|")+ "|" + " "*20 , end="")
                    producer.produce(self.topic, key=str(i), value=str(row))
                    producer.flush()
                    batch = ""


    def read_file(self, filepath):
        with open(filepath, 'r') as f:
            return f.read()

    def feed_dir(self, source_dir):
        exclude_patterns = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), source_dir)
                if file.endswith("main.tex") and not file.startswith(".") and not any(fnmatch.fnmatch(rel_path, p) for p in exclude_patterns):
                    fp = os.path.join(root, file)
                    yield fp, self.read_file(fp)

    def produce(self):
        # def ignore_interrupt(signal_number, frame):
        #     print("Ignoring KeyboardInterrupt in producer process.")
        # signal.signal(signal.SIGINT, ignore_interrupt)

        producer = Producer(self.config)
        try:
            for fname, content in self.feed_dir(self.source):
                if self.stop_event.is_set():
                    break
                #print(self.topic, str(fname), str(content))
                ts = datetime.now().strftime(r"%y.%m.%d-%H.%M.%S")
                fn = fname.split("/")[1]
                producer.produce(self.topic, key=fn + "." + ts, value=str(content))
                producer.flush()
                print('feeding file ', fn)
                time.sleep(self.interval_sec)
            print("All files in source are over.")
           
        except Exception as e:
            print(f"Error in Kafka producer: {e}")
            raise e
        finally:
            producer.flush()
            print("Producer stopped.")

    def start(self):
        self.stop_event.clear()
        self.process = Process(target=self.produce)
        self.process.start()

    def stop(self):
        self.stop_event.set()
        self.process.join()
        
        
    def reset_topic(self):
        """Deletes and recreates the Kafka topic."""
        try:
            # Delete the topic
            self.admin_client.delete_topics([self.topic])
            print(f"Topic '{self.topic}' deleted.")

            # Recreate the topic with the default configuration
            new_topic = NewTopic(self.topic, num_partitions=1, replication_factor=1)
            self.admin_client.create_topics([new_topic])
            print(f"Topic '{self.topic}' recreated.")
        except Exception as e:
            print(f"Error resetting topic '{self.topic}': {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Waveform Processing Pipeline")
    parser.add_argument("--mode", choices=["run", "reset"], default="run", help="run producer or reset topic")
    # Initialize the producer
    config = {'bootstrap.servers': "localhost:9092"}

    producer = KafkaProducer(topic=conf.articles_topic, source = 'data', config=config, interval_sec=1)
    #
    args = parser.parse_args()

    if args.mode == "run":
    # Start the producer
        producer.start()
        print(f"Producer for topic '{conf.articles_topic}' started.")
    elif args.mode == "reset":
        producer.reset_topic() 

    # try:
    #     stream_thread.join()
    # except KeyboardInterrupt:
    #     print("Shutting down gracefully...")
    #     producer.stop()
    #     # reset topic to start producing again 
    #     producer.reset_topic()
    #     print("kafka producer stopped.")