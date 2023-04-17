import argparse
import hashlib
import gzip
import json
import re
import uuid
from datetime import datetime
from typing import Dict, Union

import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--target_dir', type=str,
                    default="./data/github/processed")
args = parser.parse_args()

# Regex to strip repated copyright comment blocks
CPAT = re.compile("copyright", re.IGNORECASE)
PAT = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")


def get_timestamp() -> str:
    return datetime.now().isoformat()


def clean_copyright_comments(content: str):
    r = PAT.search(content)
    if r:
        # found one, now see if it contains "copyright", if so strip it
        span = r.span()
        sub = content[span[0]:span[1]]
        if CPAT.search(sub):
            # cut it
            content = content[: span[0]] + content[span[1]:]

        return content

    lines = content.split('\n')
    skip = 0

    # Greedy replace any file that begins with comment block, most
    # are copyright headers
    for k in range(len(lines)):
        if (
                lines[k].startswith("//") or
                lines[k].startswith("#") or
                lines[k].startswith("--") or
                not lines[k]
        ):
            skip = skip + 1
        else:
            break

    if skip:
        # we skipped, consume it
        content = "\n".join(lines[skip:])

    return content


def get_filecontent_stats(content: str) -> Dict[str, Union[int, str]]:
    # split content into lines and get line lengths
    line_lengths = list(map(len, content.splitlines()))

    if len(line_lengths) == 0:
        return {
            "line_count": 0,
            "max_line_length": 0,
            "avg_line_length": 0,
            "alnum_prop": 0,
        }

    # get max line length
    max_length = max(line_lengths)

    # get average line length
    avg_length = len(content) / len(line_lengths)

    # get proportion of alphanumeric characters
    alnum_count = sum(map(lambda char: 1 if char.isalnum() else 0, content))
    alnum_prop = alnum_count / len(content)

    return {
        "line_count": len(line_lengths),
        "max_line_length": max_length,
        "avg_line_length": avg_length,
        "alnum_prop": alnum_prop,
    }


def preprocess_source(source_fp: pathlib.Path, hash_table: dict):
    chunk_stats = []
    cleaned_records = []

    with gzip.open(source_fp, mode="rt", encoding="utf-8") as in_file:
        while True:
            jstr = in_file.readline()

            if not jstr:
                break

            result = json.loads(jstr)

            # skip pub/key certfiicates
            if result['path'].endswith(".crt"):
                continue

            if result['path'] == "LICENSE":
                continue

            # comptue hash of content
            digest = hashlib.md5(result['content'].encode('utf8')).hexdigest()

            # skip if we've seen this before
            if digest in hash_table:
                continue

            # add to hash table
            hash_table[digest] = 1

            # look for C style multi line comment blocks
            try:
                content = clean_copyright_comments(result['content'])
            except Exception as e:
                print(f"[{get_timestamp()}][ERROR] "
                      f"fp={source_fp}; "
                      f"Error cleaning copyright comments: {e}")
                continue

            # get file content stats (line count, max line length, avg line
            # length)
            try:
                file_stats = get_filecontent_stats(content)
            except Exception as e:
                print(f"[{get_timestamp()}][ERROR] "
                      f"fp={source_fp}; "
                      f"Error getting file stats: {e}")
                continue

            # add hash to file stats for later deduplication
            file_stats["content_hash"] = digest
            file_stats["path"] = result.get('path', "")
            chunk_stats.append(file_stats)

            # bring result into the right format
            record = {
                "text": content,
                "meta": {
                    "content_hash": digest,
                    "timestamp": "",
                    "source": "github",
                    "line_count": file_stats["line_count"],
                    "max_line_length": file_stats["max_line_length"],
                    "avg_line_length": file_stats["avg_line_length"],
                    "alnum_prop": file_stats["alnum_prop"],
                    **{
                        k: v for k, v in result.items() if k != "content"
                    }
                }
            }

            cleaned_records.append(record)

    return chunk_stats, cleaned_records


def main():
    flush_every = 20
    run_id = uuid.uuid4().hex

    run_fp = pathlib.Path(args.target_dir) / f"run_{run_id}.jsonl"
    stats_fp = pathlib.Path(args.target_dir) / f"stats_{run_id}.jsonl"

    print(f"[{get_timestamp()}][INFO] Writing records to {run_fp}")
    print(f"[{get_timestamp()}][INFO] Writing stats to {stats_fp}")

    stats_file = open(stats_fp, "w")
    records_file = open(run_fp, "w")

    # process list of *.gz files in input_file
    with open(args.input, "r") as input_file:
        files_to_process = input_file.readlines()

    total_files_to_process = len(files_to_process)

    hash_table = {}

    for file_num, fp in enumerate(files_to_process, start=1):
        fp = fp.strip()

        if not fp:
            print(f"[{get_timestamp()}][WARNING]"
                  f"[{file_num}/{total_files_to_process}] "
                  f"Skipping empty line {fp}")
            continue

        if not fp.endswith(".gz"):
            print(f"[{get_timestamp()}][WARNING]"
                  f"[{file_num}/{total_files_to_process}] "
                  f"Skipping {fp}")
            continue

        source_fp = pathlib.Path(fp)

        print(f"[{get_timestamp()}][INFO]"
              f"[{file_num}/{total_files_to_process}] "
              f"Processing {fp}")

        # get file stats and clean records
        chunk_stats, cleaned_records = preprocess_source(
            source_fp, hash_table
        )

        # write out stats
        for stats in chunk_stats:
            stats_file.write(json.dumps(stats) + "\n")

        # write out cleaned records
        for record in cleaned_records:
            records_file.write(json.dumps(record) + "\n")

        if file_num % flush_every == 0:
            # make sure data is written to disk
            print(f"[{get_timestamp()}][INFO] Flushing ...")
            stats_file.flush()
            records_file.flush()

    stats_file.close()
    records_file.close()


if __name__ == '__main__':
    main()
