#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output="logs/github/download/download-docs-%j.out"
#SBATCH --error="logs/github/download/download-docs-%j.err"
#SBATCH --time=23:59:00
#SBATCH --array=1-100
#SBATCH --job-name=download-github

set -e

DATA_DIR="./data/github/src"
PARTS_DIR="./data/github/partitions"

INPUT_FILE=$(ls ${PARTS_DIR}/*.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "input file is ${INPUT_FILE}"

gsutil cp $(cat $INPUT_FILE) $DATA_DIR
