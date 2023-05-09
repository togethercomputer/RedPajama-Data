#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/github/dedupe/dedupe-%j.out"
#SBATCH --error="logs/github/dedupe/dedupe-%j.err"
#SBATCH --time=23:59:00
#SBATCH --job-name=dedupe-github

set -e

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

FIRST_STEP_DIR="./data/github/processed"
TARGET_DIR="./data/github/processed_deduped"

mkdir -p logs/github/dedupe

echo "input dir is $FIRST_STEP_DIR"
echo "target dir is $TARGET_DIR"

python github_global_dedup.py --first_step_dir $FIRST_STEP_DIR --target_dir $TARGET_DIR
