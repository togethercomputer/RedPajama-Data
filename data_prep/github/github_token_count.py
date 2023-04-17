import argparse
import os
from transformers import AutoTokenizer
import json
import multiprocessing as mp
import pathlib
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=None)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")

FRACTION = 0.1


def get_timestamp() -> str:
    return datetime.now().isoformat()


def get_token_count(text):
    return len(tokenizer.tokenize(text))


def main():
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", mp.cpu_count()))

    data_fp = pathlib.Path(args.data_file)

    # get total number of records in file
    print(f"[{get_timestamp()}][INFO] Counting records in {data_fp} ...")
    with open(data_fp, "r") as f:
        num_records = sum(1 for _ in f)
    print(f"[{get_timestamp()}][INFO] Found {num_records} records.")

    print(f"[{get_timestamp()}][INFO] Loading data...")
    with open(data_fp, "r") as f:
        # get a batch of records
        records = []

        for _ in range(int(num_records * FRACTION)):
            line = f.readline()

            if not line:
                break

            try:
                record = json.loads(line)
            except json.decoder.JSONDecodeError:
                continue

            records.append(record["text"])

    print(f"[{get_timestamp()}][INFO] Start token count...")

    # count tokens in records
    with mp.Pool(num_cpus) as pool:
        token_counts = pool.map(get_token_count, records)

    total_token_count = sum(token_counts)

    result = {
        "total_token_count": total_token_count,
        "sampling_fraction": FRACTION,
        "total_count_estimate": total_token_count / FRACTION
    }

    out_fp = data_fp.parent / \
             f"{data_fp.stem.replace('deduped', 'token_count')}.json"
    with open(out_fp, mode="w") as out:
        out.write(json.dumps(result))

    print(json.dumps(result, indent=4))

    print(f"[{get_timestamp()}][INFO] Result written to {out_fp}.")
    print(f"[{get_timestamp()}][INFO] Done.")


if __name__ == '__main__':
    main()
