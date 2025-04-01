cd /home/linuxu/Michael/BigData/ArticleReader
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pyspark_TTS
export PYSPARK_DRIVER_PYTHON=python
spark-submit --master local-cluster[3,4,2048] \
--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 \
Spark/TTS.py --mode stream --kafka-topic articles --kafka-servers localhost:9092 --output-type fs --output-type parquet --output output/ 




 # ; /usr/bin/env /home/linuxu/anaconda3/envs/pyspark_TTS/bin/python /home/linuxu/.vscode/extensions/ms-python.debugpy-2025.4.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 42623 -- /home/linuxu/Michael/BigData/ArticleReader/Spark/TTS.py 