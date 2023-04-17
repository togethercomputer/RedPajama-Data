import argparse
from datetime import datetime
import json
import gzip
import os
import pathlib
import joblib
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="./data/c4/en")
parser.add_argument('--output_dir', type=str, default="./data/c4/processed_en")
parser.add_argument('--max_files', type=int, default=-1)
args = parser.parse_args()


def get_timestamp() -> str:
    return datetime.now().isoformat()


def process_record(record):
    return {
        "text": record["text"],
        "meta": {
            "timestamp": record["timestamp"],
            "url": record["url"],
            "language": "en",
            "source": "c4"
        }
    }


def process_file(fp):
    print(f"[{get_timestamp()}][INFO] start processing {fp}...")
    out_dir = pathlib.Path(args.output_dir)
    out_fp = out_dir / fp.with_suffix("").name.replace("json", "jsonl")

    with gzip.open(fp, "r") as in_f:
        records = [json.loads(line) for line in in_f.readlines()]

    with open(out_fp, "w") as out_f:
        for record in records:
            record = process_record(record)
            if record is not None:
                out_f.write(json.dumps(record) + "\n")

    print(f"[{get_timestamp()}][INFO] done processing {fp}...")


def main():
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", joblib.cpu_count()))
    print(f"Using {num_cpus} processes")

    out_dir = pathlib.Path(args.output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    records_files = list(pathlib.Path(args.data_dir).glob("*.json.gz"))
    if args.max_files > 0:
        records_files = records_files[:args.max_files]

    Parallel(n_jobs=num_cpus)(
        delayed(process_file)(fp) for fp in records_files
    )


if __name__ == '__main__':
    main()
