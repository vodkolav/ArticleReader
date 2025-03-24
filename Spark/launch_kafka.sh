cd /usr/local/kafka/kafka_2.13-3.2.1

echo starting zookeeper

bin/zookeeper-server-start.sh config/zookeeper.properties &

sleep 10
echo starting kafka
bin/kafka-server-start.sh config/server.properties &

sleep 10
read -p "press Enter to shutdown Kafka" fullname

bin/kafka-server-stop.sh 
sleep 10

bin/zookeeper-server-stop.sh
