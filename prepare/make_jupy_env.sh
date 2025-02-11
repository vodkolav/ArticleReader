#!/bin/bash

# Variables
ENV_YAML="jupy.yml"
ENV_NAME="jupy"

# If any command fails (returns a nonzero exit status), 
# the script will terminate immediately instead of continuing.
set -e

# Set conda-forge as the default channel
echo "Configuring conda-forge as the default channel..."
conda config --add channels conda-forge
conda config --set channel_priority strict


# Ensure mamba is installed
echo "Checking for mamba..."
if ! command -v mamba &> /dev/null
then
    echo "mamba not found, installing..."
    conda install -n base -c conda-forge mamba -y
fi

# Set mamba as the default solver
#echo "Configuring mamba as the default solver..."
#conda config --set solver libmamba


# Create Conda environment using mamba
echo "Creating Conda environment from $ENV_YAML using mamba..."
mamba env create -f $ENV_YAML 

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Enable Jupyter extensions
echo "Installing and enabling JupyterLab extensions..."
#jupyter labextension install @jupyterlab/git
#jupyter serverextension enable --py jupyterlab_code_formatter

# Ensure JupyterLab Git extension works
echo "Rebuilding jupyter"
jupyter lab build

# Deactivate environment
conda deactivate

echo "Done. To use Jupyter Lab, activate the environment and run:"
echo "  conda activate $ENV_NAME"
echo "  jupyter lab"