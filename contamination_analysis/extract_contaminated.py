import argparse
from datetime import datetime
import itertools
import json
import pathlib
import hashlib
import multiprocessing as mp
from typing import Set, Dict, ByteString, Tuple, List
import joblib

from src.ngrams import form_ngrams

MAX_HASH_IN_MEMORY = 10_000
VALID_EXTENSIONS = [".jsonl", ".jsonl.zst"]

HASH_CHUNK_SIZE = 1_000_000
HASH_BYTES = 20  # sha1 hash is 20 bytes
INSTANCES_FN_PATTERN = "*/instances.json"


def get_timestamp():
    return datetime.now().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", type=str,
                        default="./data/benchmark_output")
    parser.add_argument("--ngrams_dir", type=str,
                        default="./data/rpsample/splits/13grams")
    parser.add_argument("--outdir", type=str,
                        default="./data/rpsample/contaminated")
    return parser.parse_args()


def generate_test_ngrams(data_dir: pathlib.Path, nval: int):
    with mp.Pool(processes=joblib.cpu_count()) as pool:
        results: List[Tuple[str, Dict[ByteString, List[str]]]] = pool.starmap(
            parse_test_slice,
            zip(
                data_dir.rglob(INSTANCES_FN_PATTERN),
                itertools.repeat(nval)
            )
        )
    return results


def parse_test_slice(
        test_fp: pathlib.Path, nval: int
) -> Tuple[str, Dict[ByteString, List[str]]]:
    with open(test_fp, "r") as f:
        data = json.load(f)

    # contains the mapping from ngram hash to instance id and parent directory
    hashes_instances_mapping: Dict[ByteString, List[str]] = dict()

    for instance in data:
        text = instance["input"]["text"]
        instance_id = instance["id"]

        for digest in set(
                hashlib.sha1(" ".join(ngram).encode("utf-8")).digest()
                for ngram in form_ngrams(sequence=iter(text.split()), n=nval)
        ):
            hashes_instances_mapping[digest] = \
                hashes_instances_mapping.get(digest, []) + [instance_id]

    return str(test_fp), hashes_instances_mapping


def get_overlapping_ngram_hashes(
        train_hashes_fp: pathlib.Path,
        helm_hashes: Set
) -> Set:
    overlapping_hashes = set()

    print(f"[{get_timestamp()}] Detecting overlap with {train_hashes_fp}")

    with open(train_hashes_fp, "rb") as hf:
        while True:
            # read in chunks of hashes
            chunk = hf.read(HASH_CHUNK_SIZE * HASH_BYTES)

            # if no more chunks, break
            if not chunk:
                break

            # convert chunk to set of hashes
            chunk_hashes = set(
                chunk[i:i + HASH_BYTES]
                for i in range(0, len(chunk), HASH_BYTES)
            )

            # return set of hashes that overlap with helm hashes
            overlapping_hashes.update(
                chunk_hashes.intersection(helm_hashes)
            )

    print(f"[{get_timestamp()}] Done with {train_hashes_fp}; "
          f"found {len(overlapping_hashes)} overlapping ngrams")

    return overlapping_hashes


def main():
    args = parse_args()

    # parse info from ngrams
    info_fp = pathlib.Path(args.ngrams_dir) / "info.json"

    with open(info_fp, "r") as f:
        nval = json.load(f)["n"]

    print(f"nval: {nval}")

    # get benchmark input filepaths
    test_data_dir = pathlib.Path(args.test_data_dir)

    # get the test ngrams hashes: this is a list consisting of the test
    # filepaths and the mapping from ngram hash to the ids of instances that
    # contain the ngram
    test_ngrams = generate_test_ngrams(test_data_dir, nval)

    # get a set of all test ngram digests
    test_ngrams_digests = set(itertools.chain(
        *[list(ngrm_mapping.keys()) for _, ngrm_mapping in test_ngrams]
    ))

    # # get a set of all test instance ids
    test_instance_ids = set(itertools.chain(
        *[
            set(itertools.chain(*list(ngrm_mapping.values())))
            for _, ngrm_mapping in test_ngrams
        ]
    ))

    print(f"[{get_timestamp()}] total ngrams in the test set:",
          len(test_ngrams_digests))
    print(f"[{get_timestamp()}] total instances in the test set:",
          len(test_instance_ids))

    # get overlapping ngrams
    with mp.Pool(processes=joblib.cpu_count()) as pool:
        contaminated_hashes = pool.starmap(
            get_overlapping_ngram_hashes,
            zip(pathlib.Path(args.ngrams_dir).glob("*.bin"),
                itertools.repeat(test_ngrams_digests))
        )

    # get unique contaminated ngram hashes
    unique_contaminated_hashes = set.union(*contaminated_hashes)
    print(f"[{get_timestamp()}] total contaminated ngrams:",
          len(unique_contaminated_hashes))

    # save contaminated test instance ids
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # get contaminated instance ids
    contaminated_instance_ids = set()
    for test_fp, ngrm_mapping in test_ngrams:
        test_slice_name = pathlib.Path(test_fp).parent.stem
        fp_instance_ids = set(
            itertools.chain(*[
                ngrm_mapping.get(digest, [])
                for digest in unique_contaminated_hashes
            ])
        )
        contaminated_instance_ids.update(fp_instance_ids)

        with open(outdir / (test_slice_name + ".json"), "w") as f:
            json.dump(list(fp_instance_ids), f)

    print(f"[{get_timestamp()}] total contaminated instances:",
          len(contaminated_instance_ids))


if __name__ == '__main__':
    main()
