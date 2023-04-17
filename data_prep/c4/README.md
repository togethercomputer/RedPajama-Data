### C4

Follow these instructions to create the C4 dataset.

#### Setup

Install the dependencies specified in `c4_requirements.txt`:

```bash
pip install -r c4_requirements.txt
```

Then, setup the following folder structure (or create corresponding symlinks) to store data:

```bash
mkdir -p data
mkdir -p logs/c4
```


#### Data Download

We use the C4 dataset hosted on Huggingface. You will first need to install Git Large File Storage (Git LFS) on
your machine to downloaded the files (check out the instructions on https://git-lfs.com/). Once git-lfs is installed, you
can clone the repository and get the data by running the following commands from the `data` folder:

```bash
cd data/
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"
```

This will download ~300GB of (compressed) data.

#### Data Preparation

The only preprocessing step we apply is to bring the data into our own format. To that end, in `data_prep/c4` folder, you can run

```bash
python c4_reformat.py --data_dir ./data/c4/en --output_dir ./data/c4/processed_en --max_files -1
```

or, if you are on a slurm cluster, you can use our slurm script which is configured to request 64 cores and 128GB of
memory:

```bash
sbatch scripts/c4-reformat-slurm.sbatch
```