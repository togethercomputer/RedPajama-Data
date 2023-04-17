#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/github/merge-dedup/merge-dedupe-%j.out"
#SBATCH --error="logs/github/merge-dedup/merge-dedupe-%j.err"
#SBATCH --time=23:59:00
#SBATCH --array=1-100
#SBATCH --job-name=dedup-github

set -e

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

FIRST_STEP_DIR="./data/github/processed"
INPUT_FILE=$(ls ${FIRST_STEP_DIR}/run_*.jsonl | sed -n "${SLURM_ARRAY_TASK_ID}p")
TARGET_DIR="./data/github/processed_deduped"

mkdir -p logs/github/merge-dedup/
mkdir -p $TARGET_DIR

echo "input file is $FIRST_STEP_DIR"
echo "target dir is $TARGET_DIR"

python github_merge_dedup.py \
  --first_step_dir $FIRST_STEP_DIR \
  --target_dir $TARGET_DIR \
  --input $INPUT_FILE
