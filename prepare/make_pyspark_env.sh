#!/bin/bash



# Variables
ENV_YAML="pyspark_TTS.yml"
ENV_NAME="pyspark_TTS"

set -e
# Path to existing Spark installation
SPARK_HOME="/usr/local/spark/"
#JAVA_HOME="/path/to/java"


# install Spark 
# TODO: check if same version aready installed
wget https://dlcdn.apache.org/spark/spark-3.5.4/spark-3.5.4-bin-hadoop3.tgz

sudo tar xvf spark-3.5.4-bin-hadoop3.tgz -C $SPARK_HOME

sudo chmod -R 777 $SPARK_HOME
cd $SPARK_HOME
sudo unlink spark
sudo ln -sv spark-3.5.4-bin-hadoop3 spark
sudo chown -h linuxu:linuxu spark


# Create Conda environment from YAML
echo "Creating Conda environment from $ENV_YAML..."
#conda env create -v -f $ENV_YAML 

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME  

# Configure env variables to find spark and kafka
conda env config vars set PATH=$PATH:/usr/local/spark/spark/bin:/usr/local/kafka/kafka/bin
conda env config vars set SPARK_HOME=/usr/local/spark/spark
#export PATH=$PATH:/usr/local/spark/spark/bin
conda env config vars set KAFKA_HOME=/usr/local/kafka/kafka
#export PATH=$PATH:/usr/local/kafka/kafka/bin
conda env config vars set PYSPARK_DRIVER_PYTHON=ipython
#/home/linuxu/anaconda3/envs/$ENV_NAME/bin/ipython
#export PYSPARK_DRIVER_PYTHON_OPTS='lab'
conda env config vars set PYSPARK_PYTHON=/home/linuxu/anaconda3/envs/$ENV_NAME/bin/python
#export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}



# # Add environment variables to the environment activation script
# echo "Configuring environment variables..."
# ENV_ACTIVATE_SCRIPT=$(conda env list | grep $ENV_NAME | awk '{print $2}')/etc/conda/activate.d/env_vars.sh
# mkdir -p $(dirname $ENV_ACTIVATE_SCRIPT)
# echo "export SPARK_HOME=$SPARK_HOME" > $ENV_ACTIVATE_SCRIPT
# #echo "export JAVA_HOME=$JAVA_HOME" >> $ENV_ACTIVATE_SCRIPT
# echo "export PYSPARK_PYTHON=python" >> $ENV_ACTIVATE_SCRIPT
# echo "export PYSPARK_DRIVER_PYTHON=python" >> $ENV_ACTIVATE_SCRIPT



# Register Jupyter kernel
# Using virtualenv or conda envs, you can make your IPython kernel in one env available to Jupyter 
# in a different env. To do so, run ipykernel install from the kernel’s env, with –prefix pointing to the Jupyter env:
echo "Registering Jupyter kernel..."
python -m ipykernel install --name=$ENV_NAME --display-name "Python ($ENV_NAME)" --prefix=/home/linuxu/anaconda3/envs/jupy

# Deactivate environment
#conda deactivate

# create environment for analyzing results
ENV_NAME=analitic
conda env create -f $ENV_NAME 
conda activate $ENV_NAME 
python -m ipykernel install --name=$ENV_NAME --display-name "Python ($ENV_NAME)" --prefix=/home/linuxu/anaconda3/envs/jupy

echo "Done. Use Jupyter Lab and select the kernel 'Python ($ENV_NAME)'."

