import functools
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
from pathlib import Path
from tqdm import tqdm

from core.document import Document
from utilities.io import Reader


def _compute_hash_features(text: str, buckets: int):
    r""" Compute the hash features for a given text """
    # compute hash features directly in the document class
    # for consistency
    document = Document(
        content=text, domain=None, precompute_ngrams=False,
        precompute_hash_features=True, dsir_buckets=buckets
    )
    return document.hash_features, document.num_raw_words


class HashDist:
    # output file naming convention
    output_file_fmt_counts = "{dataset}.{lang}.{buckets}.counts.npy"
    output_file_fmt_lambda = "{dataset}.{lang}.lambda.npy"

    def __init__(
            self, artifacts_dir, num_samples, buckets, max_workers, logger
    ):
        self._artifacts_dir = artifacts_dir
        self._num_samples = num_samples
        self._buckets = buckets
        self._max_workers = max_workers
        self._logger = logger

    def run(self, lang, datafile, dataset):
        log_prefix = f"{self.__class__.__name__}(" \
                     f"lang={lang}, datafile={datafile}, dataset={dataset})"
        datafile = str(Path(datafile).absolute())

        out_dir = Path(self._artifacts_dir) / "dsir" / f"{lang}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp_dist = out_dir / self.output_file_fmt_counts.format(
            dataset=dataset, lang=lang, buckets=self._buckets
        )
        out_fp_lambda = out_dir / self.output_file_fmt_lambda.format(
            dataset=dataset, lang=lang
        )
        self._logger.info(
            f"{log_prefix} Start dsir computation for {lang}-{dataset}"
        )
        self._logger.info(f"{log_prefix} Reading data from {datafile}")
        self._logger.info(f"{log_prefix} Write distribution to {out_fp_dist}")
        self._logger.info(f"{log_prefix} Write lambda to {out_fp_lambda}")

        if self._max_workers is not None:
            if self._max_workers < 0:
                raise ValueError("max_workers must be >= 0")
            max_proc = min(self._max_workers, mp.cpu_count() - 1)
        else:
            max_proc = mp.cpu_count() - 1

        self._logger.info(f"{log_prefix} Using {max_proc} processes")
        reader = Reader(schema=[("text", str)])

        def _wrap_reader():
            r""" wrap reader so that it can be used with multiprocessing.
            Otherwise, pickling of records fails. """
            for record in reader.read(
                    uri="file://" + datafile,
                    max_samples=self._num_samples,
                    return_idx=False
            ):
                yield record.text

        global_dist = np.zeros(self._buckets, dtype=np.int64)

        # MLE estimator for lambda of Poisson distribution
        lambda_mle = 0
        num_samples = 0

        with Pool(max_proc) as pool:
            for dist, dlen in tqdm(
                    pool.imap_unordered(
                        functools.partial(
                            _compute_hash_features,
                            buckets=self._buckets
                        ),
                        _wrap_reader()
                    ),
                    total=self._num_samples,
                    desc=f"Reading {datafile}"
            ):
                global_dist += dist
                lambda_mle += dlen
                num_samples += 1

        # save lambda
        np.save(file=str(out_fp_lambda), arr=lambda_mle / num_samples)
        self._logger.info(f"{log_prefix} Saved lambda to {out_fp_lambda}")

        # save distribution
        np.save(file=str(out_fp_dist), arr=global_dist)
        self._logger.info(f"{log_prefix} Saved distribution to {out_fp_dist}")

        self._logger.info(f"{log_prefix} Finished dsir for {lang}-{dataset}.")
