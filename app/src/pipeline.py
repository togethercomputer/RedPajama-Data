import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime as dt
import gc
import json
import logging
import multiprocessing as mp
import numpy as np
from pathlib import Path
import random
import re
import time
from typing import Dict, List
import uuid
from urllib.parse import urlparse

from core.worker import Worker
from utilities.logging.trackers import RateTracker
from utilities.logging.mp import *


def get_timestamp():
    return dt.now().isoformat()


def monitor_progress(
        logging_queue: mp.Queue, monitor_queue: mp.Queue, languages: List[str],
        total_uri_counts: Dict[str, int]
):
    start_time = time.time()
    log_str = " | ".join([f"{lang}: {{{lang}:,}}" for lang in languages]) + \
              " | total: {total:,} | {rate:.2f} docs/s" + \
              " | {lang}: processed {uc}/{tuc} uris"

    # setup logging
    configure_worker_logger(logging_queue, level=logging.INFO)
    logger = logging.getLogger()

    total_docs = 0

    uri_counts_per_lang = {lang: 0 for lang in languages}
    doc_counts_per_lang = {k: 0 for k in languages}

    logger.info(f"Start monitoring...")

    rate_tracker = RateTracker(n=200)
    current_lang = None

    try:
        while True:

            batch_time = time.time()

            if (data := monitor_queue.get()) is None:
                break

            lang = data["lang"]
            num_docs = data["num_docs"]
            uri_complete = data.get("uri_complete", False)

            if uri_complete:
                # we received a uri complete signal -- record that one more
                # uri has been processed
                uri_counts_per_lang[lang] += 1
                continue

            if lang != current_lang:
                current_lang = lang
                rate_tracker.reset()
                logger.info(f"reset tracker for {lang}")

            doc_counts_per_lang[lang] += num_docs
            total_docs += num_docs

            rate_tracker.update(num_docs, batch_time)
            rate = rate_tracker.get_rate(time.time())

            # log stats
            logger.info(log_str.format(
                **doc_counts_per_lang, total=total_docs, rate=rate,
                lang=lang, uc=uri_counts_per_lang[lang],
                tuc=total_uri_counts[lang],
            ))

    except KeyboardInterrupt:
        logger.error(f"KeybordInterrupt. Shutting down progress monitor.")
        return

    logger.info("=" * 80)
    logger.info(f"Done. Total time {time.time() - start_time:2f}s")
    logger.info("=" * 80)
    logger.info("Document counts:")

    for lang, num_docs in doc_counts_per_lang.items():
        logger.info(f"{lang}: {num_docs:,}")

    logger.info(f"Total: {total_docs:,}")
    logger.info("=" * 80)
    logger.info(f"Progress monitor done.")


def main_logger_process(logging_queue: mp.Queue, logfile: Path):
    configure_listener_logger(logfile=logfile, level=logging.INFO)

    while True:
        message = logging_queue.get()
        if message is None:
            break
        logger = logging.getLogger(message.name)
        logger.handle(message)


class RPSignalJob:
    r""" Class for running the rp_signals pipeline """

    # descriptions for input and output arguments. This will be shown using
    # the --help flag
    input_descr = "The input must be provided as a listings file containing " \
                  "the relative paths to the data files, one per line as " \
                  "relative paths (to input_pase_uri)."

    def __init__(self):
        self._args = self.parse_arguments()
        self._job_id = str(uuid.uuid4())[:16]

        random.seed(self._args.seed)

        # convenience access to args
        self._languages = self._args.langs
        self._inputs_per_process = self._args.inputs_per_process

        # minhash
        self._minhash_ngram_size = self._args.minhash_ngram_size
        self._minhash_num_permutations = self._args.minhash_num_permutations
        self._minhash_similarities = self._args.minhash_similarities

        # artifacts
        self._artifacts_dir = Path(self._args.artifacts_dir)
        self._classifiers_dir = self._artifacts_dir / "classifiers"
        self._dsir_dir = self._artifacts_dir / "dsir"
        self._bad_words_dir = self._artifacts_dir / "bad_words"
        self._bad_urls_dir = self._artifacts_dir / "bad_urls"

        # i/o args
        self._snapshot_id = self._args.cc_snapshot_id
        self._input_listings = self.__parse_input_listings()
        self._output_base_uri = self._args.output_base_uri
        self._output_base_uri_parsed = urlparse(self._output_base_uri)
        self._log_dir = Path(
            self._output_base_uri_parsed.path
        ) / "logs" / self._snapshot_id

        # get classifier filepaths
        self._classifiers_files = self.__parse_classifiers_dir()

        # get filepaths for importance weights
        self._dsir_files = self.__parse_dsir_dir()

    def parse_arguments(self):
        if self.__doc__ is not None:
            description = " - " + self.__doc__
        else:
            description = self.__class__.__name__

        parser = argparse.ArgumentParser(
            prog=self.__class__.__name__, description=description
        )

        # input and outputs
        parser.add_argument(
            "--input", type=str, default=None, help=self.input_descr
        )
        parser.add_argument(
            "--input_base_uri", type=str,
            default=None,
            help="Base URL (prefix) used for files list in input. Used to "
                 "select the access method: s3://<path>/ or file://<path>/"
        )
        parser.add_argument(
            "--output_base_uri", type=str,
            default=None,
            help="Base URL (prefix) used for files list in output. Used to "
                 "select the access method: s3://<path>/ or file://<path>/"
        )
        parser.add_argument(
            "--filename_keep_patterns", type=str, nargs="+", default=None,
            help="list of regex patterns to match against filenames to keep"
        )
        parser.add_argument(
            "--cc_snapshot_id", type=str, default=None,
            help="id of the common crawl snapshot to process."
        )
        parser.add_argument(
            "--artifacts_dir", type=str, default=None,
            help="Path on the local filesystem to the directory containing "
                 "artifacts"
        )
        parser.add_argument(
            "--ext", type=str, default=".json.gz",
            help="File extension of input files; defaults to .json.gz"
        )
        parser.add_argument(
            "--max_docs", type=int, default=-1,
            help="maximum number of documents to process per input "
                 "file; for development purposes"
        )
        parser.add_argument(
            "--max_proc", type=int, default=None,
            help="maximum number of processes to use; default is the number "
                 "of available CPUs"
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="random seed"
        )
        parser.add_argument(
            "--endpoint_url", type=str, default=None,
            help="endpoint url where the s3 bucket is exposed. "
        )
        parser.add_argument(
            "--inputs_per_process", type=int, default=20,
            help="number of inputs to process per worker"
        )
        parser.add_argument(
            "--langs", type=str, nargs="+",
            default=["en"],
            help="subset of languages for which data files are processed."
        )

        # dsir
        parser.add_argument(
            "--dsir_buckets", type=int, default=10_000,
            help="dimension of feature vector for dsir"
        )

        # minhash
        parser.add_argument(
            "--minhash_ngram_size", type=int, default=None,
            help="ngram size for minhash"
        )
        parser.add_argument(
            "--minhash_num_permutations", type=int, default=None,
            help="number of permutations for minhash"
        )
        parser.add_argument(
            "--minhash_similarities", nargs="+", default=None, type=float,
            help="json string with minhash similarities"
        )

        return parser.parse_args()

    def __parse_input_listings(self) -> Dict[str, List[str]]:
        r""" Parse the input listing """
        if self._args.input is None:
            raise ValueError("Input argument must be provided")

        if not Path(self._args.input).exists():
            raise ValueError(f"Listings {self._args.input} does not exist")

        inputs_per_language = defaultdict(list)
        fn_regexes = list(map(
            lambda p: re.compile(p), self._args.filename_keep_patterns or []
        ))

        with open(self._args.input) as f:
            for line in f.readlines():
                listing = line.strip()

                if not listing:
                    continue

                lang = Path(listing).name.split("_")[0]

                if lang not in self._languages:
                    continue

                if len(fn_regexes) > 0:
                    if not any(p.match(listing) for p in fn_regexes):
                        continue

                inputs_per_language[lang].append(listing)

        return inputs_per_language

    def __parse_classifiers_dir(self) -> Dict[str, Dict[str, str]]:
        model_files = defaultdict(dict)

        for lang in self._languages:
            model_dir = self._classifiers_dir / lang
            if not model_dir.exists():
                continue
            for model_file in model_dir.glob("*.bin"):
                domain = model_file.stem.split(".")[0]
                model_files[lang][domain] = str(model_file)

        return model_files

    def __parse_dsir_dir(self) -> Dict[str, Dict[str, List[str]]]:
        dsir_filepaths = defaultdict(dict)

        for lang in self._languages:
            dsir_dir = self._dsir_dir / lang

            if not dsir_dir.exists():
                continue

            for counts_file in dsir_dir.glob("*.counts.npy"):
                domain = counts_file.stem.split(".")[0]
                dsir_filepaths[lang][domain] = [str(counts_file)]

            for lambda_file in dsir_dir.glob("*.lambda.npy"):
                domain = lambda_file.stem.split(".")[0]
                dsir_filepaths[lang][domain].append(str(lambda_file))

        return dsir_filepaths

    def __log_run_setup(self, logger):
        logger.info(f"logging outputs to {self._log_dir}")
        logger.info(f"job_id: {self._job_id}")
        logger.info(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")

        # log args
        for arg, val in vars(self._args).items():
            logger.info(f"{arg}: {val}")

        # logs job fields
        logger.info(f"classifier_files: \n"
                    f"{json.dumps(self._classifiers_files, indent=4)}")
        logger.info(f"dsir_files: \n"
                    f"{json.dumps(self._dsir_files, indent=4)}")

    def run(self):
        max_proc = min(mp.cpu_count(), self._args.max_proc or np.inf)

        # get total number of uris per language
        total_uri_counts = {
            lang: len(self._input_listings[lang]) for lang in self._languages
        }

        # setup logging
        log_file = Path(self._log_dir) / f"{self._job_id}.log"
        manager = mp.Manager()

        queue_buffer_size = 128 * (self._args.max_proc or mp.cpu_count())

        # kick off logger process
        logging_queue = manager.Queue(maxsize=queue_buffer_size)
        logger_proc = mp.Process(
            target=main_logger_process, args=(logging_queue, log_file)
        )
        logger_proc.start()

        # start progress monitor
        monitor_queue = manager.Queue(maxsize=queue_buffer_size)
        monitor_proc = mp.Process(
            target=monitor_progress,
            args=(logging_queue, monitor_queue, self._languages,
                  total_uri_counts)
        )
        monitor_proc.start()

        configure_worker_logger(queue=logging_queue, level=logging.INFO)
        logger = logging.getLogger()

        # log run setup
        self.__log_run_setup(logger)
        for lang in self._languages:
            logger.info(f"{lang}: {len(self._input_listings[lang]):,} inputs")

        for lang in self._languages:
            lang_inputs = self._input_listings[lang]
            random.shuffle(lang_inputs)

            logger.info("*" * 80)
            logger.info(f"Start processing {lang}")

            chunk_size = self._args.inputs_per_process
            input_chunks = [
                lang_inputs[i * chunk_size:(i + 1) * chunk_size]
                for i in range(len(lang_inputs) // chunk_size)
            ]

            max_docs_per_chunk = self._args.max_docs // len(input_chunks)

            with ProcessPoolExecutor(max_workers=max_proc - 2) as executor:
                futures = [
                    executor.submit(
                        self._run_chunk,
                        input_listings=chunk,
                        lang=lang,
                        max_docs=max_docs_per_chunk,
                        monitor_queue=monitor_queue,
                        logging_queue=logging_queue,
                    )
                    for chunk in input_chunks
                ]

                try:
                    for future in as_completed(futures):
                        result = future.result()
                        futures.remove(future)
                        wid = result["job_id"]
                        exc = result["exception"]
                        if exc is not None:
                            logger.error(f"__WORKER_FAIL__ ({wid}) exc={exc}")
                            continue

                        logger.info(f"__WORKER_COMPLETED__ {wid} completed.")
                except KeyboardInterrupt:
                    logger.error(f"KeyboardInterrupt. Shutting down.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

            gc.collect()

        # signal monitor to stop
        monitor_queue.put(None)
        monitor_proc.join()

        # signal logger to stop
        logging_queue.put_nowait(None)
        logger_proc.join()

        manager.shutdown()

    def _run_chunk(
            self, input_listings, lang, max_docs, monitor_queue, logging_queue
    ):

        if len(input_listings) == 0:
            return {"exception": None, "lang": lang, "job_id": None}

        proc = Worker(
            language=lang,
            snapshot_id=self._snapshot_id,
            input_listings=input_listings,
            input_base_uri=self._args.input_base_uri,
            output_base_uri=self._output_base_uri,
            log_dir=self._log_dir,
            classifier_files=self._classifiers_files.get(lang, {}),
            dsir_files=self._dsir_files.get(lang, {}),
            dsir_bucket=self._args.dsir_buckets,
            ldnoobw_dir=self._bad_words_dir,
            ut1_dir=self._bad_urls_dir,
            minhash_similarities=self._minhash_similarities,
            minhash_ngram_size=self._minhash_ngram_size,
            minhash_num_permutations=self._minhash_num_permutations,
            logging_queue=logging_queue,
            monitor_queue=monitor_queue,
            endpoint_url=self._args.endpoint_url,
            max_docs=max_docs,
            seed=self._args.seed,
            flush_interval=5000
        )

        try:
            proc.run()
            exc = None
        except Exception as e:
            exc = f"{e.__class__.__name__}: {e}"

        gc.collect()

        return {"exception": exc, "lang": lang, "job_id": proc.job_id}


if __name__ == '__main__':
    mp.set_start_method('fork')
    mp.set_executable("python")
    job = RPSignalJob()
    job.run()
