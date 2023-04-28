#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --output="logs/contamination_analysis/%j.out"
#SBATCH --error="logs/contamination_analysis/%j.err"
#SBATCH --time=23:59:00
#SBATCH --job-name=decontamination

set -e

# load modules
module load gcc/6.3.0 python/3.8.5

source .venv/bin/activate

python extract_contaminated.py \
  --test_data_dir ./data/helm/benchmark_output \
  --ngrams_dir ./data/stackexchange/splits/13grams \
  --outdir ./data/stackexchange/contaminated_helm
