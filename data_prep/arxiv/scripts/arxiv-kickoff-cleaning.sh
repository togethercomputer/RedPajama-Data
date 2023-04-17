#!/bin/bash

set -e

WORKERS=100

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

export DATA_DIR="./data/arxiv/src"
export TARGET_DIR="./data/arxiv/processed"
export WORK_DIR="./work"

mkdir -p logs/arxiv/cleaning

# setup partitions
python run_clean.py --data_dir "$DATA_DIR" --target_dir "$TARGET_DIR" --workers $WORKERS --setup

# run download in job array
sbatch scripts/arxiv-clean-slurm.sbatch
