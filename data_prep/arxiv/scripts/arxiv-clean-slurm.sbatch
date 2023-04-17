#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/arxiv/clean/clean-%j.out"
#SBATCH --error="logs/arxiv/clean/clean-%j.err"
#SBATCH --time=23:59:00
#SBATCH --array=1-100
#SBATCH --job-name=cleaning-arxiv

set -e

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

mkdir -p logs/arxiv/cleaning

export WORK_DIR="./work"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
export SLURM_CPUS_PER_TASK

INPUT_FILE=$(ls ${TARGET_DIR}/partitions/*.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "input file is ${INPUT_FILE}"

python run_clean.py \
  --data_dir "$DATA_DIR" \
  --target_dir "$TARGET_DIR" \
  --input "$INPUT_FILE" \
  --max_files -1
