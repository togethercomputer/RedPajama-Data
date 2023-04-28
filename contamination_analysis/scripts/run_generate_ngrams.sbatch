#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --output="logs/contamination_analysis/%j.out"
#SBATCH --error="logs/contamination_analysis/%j.err"
#SBATCH --time=23:59:00
#SBATCH --job-name=decontamination

set -e

# load modules
module load gcc/6.3.0 python/3.8.5

source .venv/bin/activate

python generate_ngrams.py \
  --data_dir ./data/stackexchange/splits \
  --n 13
