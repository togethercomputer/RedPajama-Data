import argparse
from datetime import datetime
import json
import multiprocessing as mp
import os
import gzip
from transformers import AutoTokenizer

import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default=None)
parser.add_argument('--target_dir', type=str, default=None)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")

extensions_whitelist = (".asm", ".bat", ".cmd", ".c", ".h", ".cs", ".cpp",
                        ".hpp", ".c++", ".h++", ".cc", ".hh", ".C", ".H",
                        ".cmake", ".css", ".dockerfile", ".f90", ".f", ".f03",
                        ".f08", ".f77", ".f95", ".for", ".fpp", ".go", ".hs",
                        ".html", ".java", ".js", ".jl", ".lua", ".md",
                        ".markdown", ".php", ".php3", ".php4", ".php5",
                        ".phps", ".phpt", ".pl", ".pm", ".pod", ".perl",
                        ".ps1", ".psd1", ".psm1", ".py", ".rb", ".rs", ".sql",
                        ".scala", ".sh", ".bash", ".command", ".zsh", ".ts",
                        ".tsx", ".tex", ".vb", "Dockerfile", "Makefile",
                        ".xml", ".rst", ".m", ".smali")


def get_token_count(text):
    token_count = len(tokenizer.tokenize(text))
    return token_count


def get_timestamp() -> str:
    return datetime.now().isoformat()


def discard_record(record):
    """ return True if we discard the record """

    text = record["text"]
    metadata = record["meta"]

    # discard empty records
    if len(text) == 0:
        return True

    # discard all records that are not whitelisted
    if not metadata["path"].endswith(extensions_whitelist):
        return True

    # discard files whose maximum line length is greater than 1000
    if metadata["max_line_length"] > 1000:
        return True

    # discard files whose average line length is greater than 100
    if metadata["avg_line_length"] > 100:
        return True

    # discard files whose proportion of alphanumeric characters is less than
    # 0.25
    if metadata["alnum_prop"] < 0.25:
        return True

    num_tokens = get_token_count(text)
    num_alpha = len([c for c in text if c.isalpha()])
    if num_alpha / num_tokens < 1.5:
        return True

    return False


def filter_line(line):
    try:
        record = json.loads(line)
    except json.decoder.JSONDecodeError:
        return None

    if discard_record(record):
        return None

    return line


def process_lines_batch(lines_batch, out_file, num_cpus):
    if len(lines_batch) == 0:
        return

    with mp.Pool(processes=num_cpus - 1) as pool:
        filtered_lines = pool.map(filter_line, lines_batch)

    for line in filtered_lines:
        if line is not None:
            out_file.write(line)

    out_file.flush()


def main():
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    batch_size = num_cpus * 5_000
    input_fp = pathlib.Path(args.data_file)
    target_dir = pathlib.Path(args.target_dir)

    output_fp = target_dir / input_fp.name.replace("deduped_", "filtered_")
    output_fp = output_fp.with_suffix(".jsonl.gz")

    print(f"[{get_timestamp()}][INFO] Processing {input_fp}")
    print(f"[{get_timestamp()}][INFO] Writing to {output_fp}")

    out_file = gzip.open(output_fp, "wt", encoding="utf-8")

    try:
        with open(input_fp, "r") as in_file:
            while True:
                lines_batch = []

                # accumulate batch
                while True:
                    line = in_file.readline()

                    if not line:
                        raise StopIteration

                    lines_batch.append(line)

                    if len(lines_batch) == batch_size:
                        break

                process_lines_batch(lines_batch, out_file, num_cpus)

    except StopIteration:
        process_lines_batch(lines_batch, out_file, num_cpus)

    out_file.close()


if __name__ == "__main__":
    main()
