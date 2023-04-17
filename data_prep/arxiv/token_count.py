import argparse
import os
from collections import defaultdict
from datetime import datetime
from transformers import AutoTokenizer
import json
import multiprocessing as mp
import pathlib
import pandas as pd
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="./data/arxiv/processed")
parser.add_argument('--max_files', type=int, default=-1,
                    help="max lines to process; this is useful for testing")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-6.9b-deduped",
)


def get_token_count(text):
    return len(tokenizer.tokenize(text))


def process_record(record):
    token_count = get_token_count(text=record["text"])
    year = record["meta"]["yymm"][:2]
    return token_count, year


def get_timestamp() -> str:
    return datetime.now().isoformat()


def print_stats(token_count_data):
    df = pd.DataFrame.from_dict(
        token_count_data, orient="index"
    )
    df = df.reset_index()
    df.columns = ["year", "count"]
    df = df.set_index("year")
    df["count"] = df["count"].astype(int)
    df["count"] = df["count"] / 1e12
    df = df.sort_values(by="count", ascending=False)
    df.loc['Total'] = df.sum(numeric_only=True)
    print(tabulate(
        df, headers=["year", "count (T)"], tablefmt="github", floatfmt=".4f"
    ))


def main():
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    print(f"Using {num_cpus} workers")

    files_processed = 0
    token_count_data = defaultdict(int)

    for filenum, fp in enumerate(pathlib.Path(args.data_dir).glob("*.jsonl")):
        with open(fp, "r") as f:
            records = [json.loads(rec) for rec in f.readlines()]

        with mp.Pool(processes=num_cpus - 2) as pool:
            results = pool.map(process_record, records)

        for counts, year in results:
            token_count_data[year] += int(counts)

        total_tokens = sum(token_count_data.values())
        print(f"[{get_timestamp()}][INFO] "
              f"processed {filenum} files; "
              f"total tokens: {total_tokens}")

        if files_processed > args.max_files > 0:
            print(f"[{get_timestamp()}][INFO] "
                  f"reached max lines")
            break

    print(json.dumps(token_count_data, indent=4))
    print(f"Total tokens: {sum(token_count_data.values())}")
    print("\n" + "=" * 80 + "\n")
    print_stats(token_count_data)


if __name__ == '__main__':
    main()
