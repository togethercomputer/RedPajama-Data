#!/bin/bash

set -e

WORKERS=100

export DATA_DIR="./data/arxiv"

mkdir -p logs/arxiv/slurm

# setup partitions
python run_download.py --aws_config aws_config.ini --workers $WORKERS --target_dir $DATA_DIR --setup

# run download in job array
sbatch scripts/arxiv-downlaod-slurm.sbatch
