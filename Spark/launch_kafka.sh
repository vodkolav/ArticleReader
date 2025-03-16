cd /usr/local/kafka/kafka_2.13-3.2.1

bin/zookeeper-server-start.sh config/zookeeper.properties &
bin/kafka-server-start.sh config/server.properties &