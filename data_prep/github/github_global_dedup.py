import argparse
import json
from datetime import datetime
from typing import Dict

import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--first_step_dir', type=str, default=None)
parser.add_argument('--target_dir', type=str, default=None)
args = parser.parse_args()


def get_timestamp() -> str:
    return datetime.now().isoformat()


def process_stats_file(source_fp: pathlib.Path, hash_table: Dict[str, str]):
    deduped_stats = []
    deduped_hashes = []

    with open(source_fp, mode="r") as in_file:
        while True:
            jstr = in_file.readline()

            if not jstr:
                break

            record_stats = json.loads(jstr)
            content_hash = record_stats["content_hash"]

            if content_hash in hash_table:
                # skip this record since it's a duplicate
                continue

            hash_table[content_hash] = content_hash
            deduped_stats.append(record_stats)
            deduped_hashes.append(content_hash)

    return hash_table, deduped_stats, deduped_hashes


def main():
    first_step_dir = pathlib.Path(args.first_step_dir)
    deduped_stats_fp = pathlib.Path(args.target_dir) / "stats_deduped.jsonl"

    print(f"[{get_timestamp()}][INFO] Deduplicating "
          f"records from {first_step_dir}")

    # get list of stats files
    stats_filepaths = list(first_step_dir.glob("stats_*.jsonl"))
    total_files_to_process = len(stats_filepaths)

    deduped_stats_file = open(deduped_stats_fp, "w")

    hash_set = {}

    for file_num, fp in enumerate(stats_filepaths, start=1):
        print(f"[{get_timestamp()}][INFO]"
              f"[{file_num}/{total_files_to_process}] "
              f"Processing {fp}")

        hash_set, deduped_stats, deduped_hashes = process_stats_file(
            fp, hash_set
        )

        # write out stats
        for stats in deduped_stats:
            deduped_stats_file.write(json.dumps(stats) + "\n")

        # write out jsonl to hashes
        out_fn = fp.name.replace("stats_", "hashes_")
        with open(pathlib.Path(args.target_dir) / out_fn, "w") as f:
            f.write(json.dumps({"hashes": deduped_hashes}) + "\n")

        print(f"[{get_timestamp()}][INFO] Flushing ...")
        deduped_stats_file.flush()

    deduped_stats_file.close()

    print(f"[{get_timestamp()}][INFO] "
          f"Total number of unique records: {len(hash_set)}")


if __name__ == '__main__':
    main()
