#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/c4/reformat/reformat-%j.out"
#SBATCH --error="logs/c4/reformat/reformat-%j.err"
#SBATCH --time=23:59:00
#SBATCH --job-name=reformat-c4

set -e

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

mkdir -p logs/c4/reformat/

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
export SLURM_CPUS_PER_TASK

python c4_reformat.py --data_dir ./data/c4/en --output_dir ./data/c4/processed_en --max_files -1
