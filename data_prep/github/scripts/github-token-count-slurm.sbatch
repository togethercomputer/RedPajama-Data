#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/github/token_count/token-count-github-%j.out"
#SBATCH --error="logs/github/token_count/token-count-github-%j.err"
#SBATCH --array=1-100
#SBATCH --time=23:59:00
#SBATCH --job-name=token-count

set -e

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

mkdir -p logs/arxiv/token_count

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
export SLURM_CPUS_PER_TASK

TARGET_DIR="./data/github_scratch/processed_deduped"
INPUT_FILE=$(ls ${TARGET_DIR}/deduped_*.jsonl | sed -n "${SLURM_ARRAY_TASK_ID}p")

echo "input file is $INPUT_FILE"

python github_token_count.py --data_file $INPUT_FILE
