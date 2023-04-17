#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/github/cleaning/cleaning-%j.out"
#SBATCH --error="logs/github/cleaning/cleaning-%j.err"
#SBATCH --time=23:59:00
#SBATCH --array=1-100
#SBATCH --job-name=cleaning-github

set -e

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

mkdir -p logs/github/cleaning/

PARTS_DIR="./data/github/processed/partitions"
INPUT_FILE=$(ls ${PARTS_DIR}/*.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
TARGET_DIR="./data/github/processed"
echo "input file is $INPUT_FILE"

python github_clean_dedup_local.py --input $INPUT_FILE --target_dir $TARGET_DIR
