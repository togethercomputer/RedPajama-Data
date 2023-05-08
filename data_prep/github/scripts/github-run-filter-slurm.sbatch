#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/github/filter/filter-%j.out"
#SBATCH --error="logs/github/filter/filter-%j.err"
#SBATCH --time=23:59:00
#SBATCH --array=1-100
#SBATCH --job-name=filter-github

set -e

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

DEDUPED_DIR="./data/github/processed_deduped/"
INPUT_FILE=$(ls ${DEDUPED_DIR}/deduped_*.jsonl | sed -n "${SLURM_ARRAY_TASK_ID}p")
TARGET_DIR="./data/github/processed_filtered"

mkdir -p logs/github/filter/
mkdir -p $TARGET_DIR

echo "input file is $FIRST_STEP_DIR"
echo "target dir is $TARGET_DIR"

python github_run_filter.py --data_file $INPUT_FILE --target_dir $TARGET_DIR
