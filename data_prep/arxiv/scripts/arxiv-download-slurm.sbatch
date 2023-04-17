#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --output="logs/arxiv/download-docs-%j.out"
#SBATCH --error="logs/arxiv/download-docs-%j.err"
#SBATCH --time=23:59:00
#SBATCH --array=1-100
#SBATCH --job-name=download-arxiv

set -e

# load modules
# <PLACEHOLDER to load modules for specific slurm cluster>

INPUT_FILE=$(ls ${DATA_DIR}/partitions/*.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "input file is ${INPUT_FILE}"

python run_download.py --aws_config aws_config.ini --target_dir $DATA_DIR --input $INPUT_FILE
