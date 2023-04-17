import argparse
import json
from datetime import datetime

import pathlib

parser = argparse.ArgumentParser()
parser.add_argument(
    '--first_step_dir', type=str,
    default="./data/github/processed_v3"
)
parser.add_argument(
    '--input', type=str,
    default="data/github/processed_v3/run_ce60fbbc14684ed8b659054801e419c8.jsonl"
)
parser.add_argument(
    '--target_dir', type=str,
    default="./data/github/processed_v3_deduped"
)
args = parser.parse_args()


def get_timestamp() -> str:
    return datetime.now().isoformat()


def main():
    input_fp = pathlib.Path(args.input)
    target_dir = pathlib.Path(args.target_dir)
    output_fp = target_dir / input_fp.name.replace("run_", "deduped_")

    # load hashes into memory
    hashes_fp = target_dir / input_fp.name.replace("run_", "hashes_")
    with open(hashes_fp) as hf:
        globally_unique_hashes = hf.readlines()[0]

    globally_unique_hashes = set(json.loads(globally_unique_hashes)["hashes"])
    output_file = open(output_fp, "w")

    print(f"[{get_timestamp()}][INFO]"
          f" Processing {input_fp}")
    print(f"[{get_timestamp()}][INFO]"
          f" Writing to {output_fp}")
    print(f"[{get_timestamp()}][INFO]"
          f" Using hashes from {hashes_fp}")

    nrecs = 0

    with open(input_fp, "r") as in_file:
        while True:
            jstr = in_file.readline()

            if not jstr:
                break

            record = json.loads(jstr)
            content_hash = record["meta"]["content_hash"]

            if content_hash not in globally_unique_hashes:
                continue

            # write to output file
            output_file.write(json.dumps(record) + "\n")
            nrecs += 1

    output_file.close()

    print(f"[{get_timestamp()}][INFO]"
          f" Processed {nrecs} records")


if __name__ == "__main__":
    main()
