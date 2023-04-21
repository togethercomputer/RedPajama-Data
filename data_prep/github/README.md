### Github

Follow these instructions to create the GitHub dataset.

We use the public GitHub dataset available on Google BigQuery. We only keep
projects that are distributed under the MIT, BSD, or Apache license. After downloading the data, we
implement further cleaning steps based on different heuristics.

#### Setup

Install the dependencies specified in `github_requirements.txt`:

```bash
pip install -r github_requirements.txt
```

Setup the following folder structure (or create corresponding symlinks) to store data:

```bash
mkdir -p data/github
mkdir -p logs/github
mkdir -p work
```

#### Data Download

Before running the query, make sure to set a destination table and dataset for the query results, and to check the
"Allow large results" tickbox in the Query settings. Once this is configured, run the following query:

```sql
SELECT *
FROM (SELECT *
      FROM `bigquery-public-data.github_repos.contents`
               INNER JOIN `bigquery-public-data.github_repos.files` USING (id))
         INNER JOIN `bigquery-public-data.github_repos.licenses` USING (repo_name)
         INNER JOIN `bigquery-public-data.github_repos.languages` USING (repo_name
    )
WHERE (license LIKE 'mit%'
    OR license LIKE 'bsd%'
    OR license LIKE 'apache%')
  AND content IS NOT NULL 
```

The resulting table will then be saved at the location you have specified in the settings. After that, export the table
to GCS, where you can later process it or download it to your machine for further processing.

Note that our processing scripts expect the data to be exported in gzip compressed jsonl files, with the prefix set to
`github_`.

To download the resulting files from gcp you can use `gsutil` and run the following command:

```bash
gsutil -m cp -r 'gs://<bucket_name>/github_*.gz'
```

where `<bucket_name>` corresponds to the bucket where you exported the data.
Alternatively, we also provide a script to download the data from GCP to your local machine in a distributed manner.
To do that, set the `GCS_BUCKET` variable in `scripts/github-prepare-download.sh` and run the following
commands:

```bash
bash scripts/github-prepare-download.sh
sbatch scripts/github-download-slurm.sbatch
```

The script `github-prepare-download.sh` first queries the GCP project to get the list of files to download and then
partitions it into 100 chunks. Each chunk is saved in a text file in `data/github/partitions`. The script
`github-download-slurm.sbatch` uses slurm job arrays to download the files in parallel with 100 workers.

#### Data Preparation

The following steps are used to prepare the data for training, and assumes that you have downloaded the data to
`data/github/src`. We first split the data into 100 chunks, and then deduplicate each chunk separately, in parallel.
The second step is to merge all chunks and perform a global deduplication. Finally, we filter out all low quality files.

*Step 1 (local deduplication and cleaning):* This step of the pipeline is mainly implemented in the script
`github_clean_dedup_local.py`. To enable parallel processing with slurm, we first split the data into 100 chunks and
save the filenames into separate text files:

```bash
bash scripts/github-prepare-local-dedup.sh
```

These can then be processed in parallel using slurm:

```bash
sbatch scripts/github-local-dedup-slurm.sbatch
```

which calls `github_clean_dedup_local.py` on the files in `data/github/processed/partitions/chunk_*.txt`.

*Step 2 (global deduplication and merging):* In the second step, we merge all chunks and perform a global deduplication by
creating a hash-set for all deduplicated chunks from the previous step. This is implemented in the script
`github_global_dedup.py` and can be run using the slurm script:

```bash
sbatch scripts/github-global-dedup-slurm.sbatch
```

For each chunk processed in step 1, this will create a file that contains all the hashes of the files in the chunk which
are globally unique. These files are then used in the final deduplication step to remove duplicates across all chunks,
using 1 worker per chunk:

```bash
sbatch scripts/github-merge-dedup-slurm.sbatch
```

This calls the script `github_merge_dedup.py` and will save all deduplicated records in jsonl files under
`./data/github/processed_deduped`. Each line in the resulting jsonl file is a json object with a field "text" containing
the text of the file and a field "meta" containing the metadata of the file, which includes the maximum line length, the
average line length, the proportion of alphanumeric characters, the number of lines and other information. This metadata
will be used in the next step to filter out low quality files.

*Step 3 (filtering):* We implemented the following set of filters to remove low quality files:

- Files with a maximum line length of more than 1000 characters
- Files with an average line length of more than 100 characters
- Files with a proportion of alphanumeric characters of less than 0.25
- Files with a ratio between the number of alphabetical characters and the number tokens of less than 1.5
- Any files whose extensions are not in the following set of whitelisted extensions:

```python
".asm", ".bat", ".cmd", ".c", ".h", ".cs", ".cpp", ".hpp", ".c++", ".h++", ".cc", ".hh", ".C", ".H", ".cmake", ".css",
".dockerfile", ".f90", ".f", ".f03", ".f08", ".f77", ".f95", ".for", ".fpp", ".go", ".hs", ".html", ".java", ".js",
".jl", ".lua", ".md", ".markdown", ".php", ".php3", ".php4", ".php5", ".phps", ".phpt", ".pl", ".pm", ".pod", ".perl",
".ps1", ".psd1", ".psm1", ".py", ".rb", ".rs", ".sql", ".scala", ".sh", ".bash", ".command", ".zsh", ".ts", ".tsx",
".tex", ".vb", "Dockerfile", "Makefile", ".xml", ".rst", ".m", ".smali"
```

The filtering is implemented in the script `github_run_filter.py` and can be run in parallel with 100 workers using the
slurm script:

```bash
sbatch scripts/github-filter-slurm.sbatch
```

This will create 100 gzip compressed jsonl files under `./data/github/processed_filtered`, containing the final GitHub
data.
