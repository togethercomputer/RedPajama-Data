#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/arxiv/token_count/token-count-arxiv-%j.out"
#SBATCH --error="logs/arxiv/token_count/token-count-arxiv-%j.err"
#SBATCH --time=23:59:00
#SBATCH --job-name=token-count

set -e

mkdir -p logs/arxiv/token_count

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
export SLURM_CPUS_PER_TASK

python token_count.py --data_dir ./data/arxiv/processed --max_files -1
