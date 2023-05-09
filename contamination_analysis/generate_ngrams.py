import argparse
from datetime import datetime
import itertools
import json
import pathlib
import string
import hashlib
import multiprocessing as mp
import joblib
import time

from src.reader import Reader
from src.ngrams import form_ngrams

VALID_EXTENSIONS = [".jsonl", ".jsonl.zst"]


def get_timestamp():
    return datetime.now().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, help="path to directory with training data",
        default="./data/rpsample/splits"
    )
    parser.add_argument(
        "--max_hash_in_mem", type=int, help="max hashes to keep in memory",
        default=500_000
    )
    parser.add_argument(
        "--n", type=int, help="ngram length",
        default=13
    )
    return parser.parse_args()


def extract_ngrams(
        input_file: pathlib.Path,
        outdir: pathlib.Path,
        n: int,
        max_hash_in_mem: int
):
    r""" extracts all ngrams from from the text in the input jsonl file and
    writes the hashes to a binary file in the output directory.

    @param input_file: path to input file
    @param outdir: path to output directory
    @param n: ngram length
    @param max_hash_in_mem: max number of hashes to keep in memory before
        writing to disk
    """
    # create table to remove punctuation and translate to lowercase
    translation_table = str.maketrans(
        # These characters
        string.ascii_lowercase + string.ascii_uppercase,
        # Become these characters
        string.ascii_lowercase * 2,
        # These are deleted
        string.punctuation
    )

    # stream input file
    reader = Reader(input_files=input_file)

    print(f"[{get_timestamp()}] Start processing {input_file}")

    # output file
    out_fp = outdir / (input_file.stem + f".{n}grams.bin")

    all_ngrams = set()

    for text in reader.read(text_key="text", yield_meta=False):
        text = text.translate(translation_table)

        # add hashes of ngrams to set
        all_ngrams.update(
            hashlib.sha1(" ".join(ngram).encode("utf-8")).digest()
            for ngram in form_ngrams(iter(text.split()), n)
        )

        if len(all_ngrams) > max_hash_in_mem:
            with open(out_fp, mode="ab") as f:
                f.write(b"".join(all_ngrams))

            all_ngrams = set()

    if len(all_ngrams) > 0:
        with open(out_fp, mode="ab") as f:
            f.write(b"".join(all_ngrams))

    print(f"[{get_timestamp()}] Done processing {input_file}")


def main():
    args = parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir = data_dir / f"{args.n}grams"

    print(f"[{get_timestamp()}] writing {args.n}gram hashes to {out_dir}")

    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists")

    out_dir.mkdir(parents=True, exist_ok=False)

    # get input files
    input_files = list(itertools.chain(
        *[list(data_dir.glob("*" + ext)) for ext in VALID_EXTENSIONS]
    ))
    n_files = len(input_files)

    # use joblib here, as mp.cpu_count() detects all cpu's (not just available
    # ones)
    n_cpus = joblib.cpu_count()

    # write info to file
    with open(out_dir / "info.json", mode="w") as f:
        f.write(json.dumps({
            "n": args.n
        }))

    t0 = time.time()

    print(f"[{get_timestamp()}] number of input files: {n_files}")
    print(f"[{get_timestamp()}] number of CPUs: {n_cpus}")
    print(f"[{get_timestamp()}] start generating {args.n}-grams")

    with mp.Pool(processes=n_cpus) as pool:
        _ = pool.starmap(
            extract_ngrams,
            zip(input_files,
                [out_dir] * n_files,
                [int(args.n)] * n_files,
                [int(args.max_hash_in_mem)] * n_files)
        )

    print(f"[{get_timestamp()}] time taken: {time.time() - t0} seconds")


if __name__ == '__main__':
    main()
