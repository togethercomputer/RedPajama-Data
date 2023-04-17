import argparse
import os
import uuid

import numpy as np
import pathlib
import tempfile
from typing import List
import joblib

from arxiv_cleaner import ArxivCleaner

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="./data/arxiv/src")
parser.add_argument('--target_dir', type=str, default="./data/arxiv/processed")
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--input', type=str, default=None,
                    help='input file from which to read keys. '
                         'This is only used when running on slurm.')
parser.add_argument('--local', action='store_true')
parser.add_argument('--setup', action='store_true',
                    help='if set, we partition the keys and into chunks.')
parser.add_argument('--max_files', type=int, default=-1,
                    help='max files to download, useful for testing')
args = parser.parse_args()

WORK_DIR = os.getenv('WORK_DIR', pathlib.Path(__file__).parent / "work")
WORK_DIR = pathlib.Path(WORK_DIR)

if not WORK_DIR.exists():
    WORK_DIR.mkdir()
    print(f"Created work directory {WORK_DIR}")


def run_clean(
        data_dir: pathlib.Path,
        target_dir: pathlib.Path,
        input_file: pathlib.Path = None,
        max_files: int = -1,
):
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", joblib.cpu_count()))
    print(f"Using {num_cpus} processes")

    worker_id = os.getenv('SLURM_ARRAY_TASK_ID', None)
    if worker_id is None:
        worker_id = str(uuid.uuid4())

    # create temporary work directory
    work_dir = pathlib.Path(
        tempfile.mkdtemp(dir=WORK_DIR, prefix=worker_id + "_")
    )

    if input_file is not None:
        # we are running on slurm
        assert input_file.exists()
        with open(input_file, 'r') as f:
            tar_fp_list = f.read().splitlines()
    else:
        tar_fp_list = None

    # create cleaner
    arxiv_cleaner = ArxivCleaner(
        data_dir=data_dir, work_dir=work_dir, target_dir=target_dir,
        worker_id=worker_id
    )

    arxiv_cleaner.run_parallel(
        max_files=max_files, tar_fp_list=tar_fp_list
    )


def partition_tar_files(
        data_dir: pathlib.Path, workers: int
) -> List[List[str]]:
    return np.array_split(
        list(str(fp) for fp in data_dir.glob('*.tar')),
        indices_or_sections=workers
    )


def main():
    # create target directory where we store the processed data
    target_dir = pathlib.Path(args.target_dir)
    if not target_dir.exists():
        target_dir.mkdir()

    data_dir = pathlib.Path(args.data_dir)
    assert data_dir.exists()

    if not args.local and not args.setup:
        # here we only download the files; this requires that setup has already
        # been run
        run_clean(
            data_dir=data_dir,
            target_dir=target_dir,
            input_file=pathlib.Path(args.input),
            max_files=args.max_files
        )
        return

    if args.setup:
        parts = partition_tar_files(data_dir=data_dir, workers=args.workers)

        if not (target_dir / "partitions").exists():
            (target_dir / "partitions").mkdir()

        for i, part in enumerate(parts):
            with open(
                    target_dir / "partitions" / f'tars_part_{i}.txt', 'w'
            ) as f:
                f.write('\n'.join(part))
        return

    # run locally; here we don't partition the tar files as slurm is not used
    if args.local:
        run_clean(
            data_dir=pathlib.Path(args.data_dir),
            target_dir=pathlib.Path(args.target_dir),
            input_file=None,
            max_files=args.max_files
        )


if __name__ == '__main__':
    main()
