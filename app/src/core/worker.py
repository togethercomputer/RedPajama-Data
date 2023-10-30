import sys
import fasttext
import gc
import hashlib
import logging
import logging.handlers
import multiprocessing as mp
import os
from pathlib import Path
import re
from typing import List, Dict, Callable, Optional
from urllib.parse import urlparse
import urllib3
import pyarrow as pa
import uuid

from core.document import Document
from core.quality_signals.content import register_content_callables
from core.quality_signals.lines import register_lines_callables
from core.quality_signals.natural_language import \
    register_natural_language_callables
from core.quality_signals.repetitions import register_repetitions_callables
from core.quality_signals.classifiers import register_classifier_callables
from core.quality_signals.importance_weights import \
    register_importance_weights_callables
from core.data_types import InputSpec
from core.schema.rp import RP_SIGNAL_SCHEMA
from dedupe.minhash import MinHash
from utilities.io import Reader, Writer, ParquetBatchWriter
from utilities.io.s3 import init_client
from utilities.logging.mp import configure_worker_logger

# disable warnings
fasttext.FastText.eprint = lambda x: None
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # noqa

_BYTE_ORDER = sys.byteorder


def _ccnet_bucket_to_int(bucket: str) -> Optional[float]:
    r""" ccnet bucket name to float mapping """
    if bucket == "head":
        return 0.0
    elif bucket == "middle":
        return 1.0
    elif bucket == "tail":
        return 2.0
    else:
        return None


class Worker:
    # output file pattern
    shard_pattern_signals = "{shard_id}.signals.json.gz"
    shard_pattern_minhash = "{shard_id}.minhash.parquet"

    # regex to extract snapshot id from uri
    snapsh_re = re.compile(r'\b\d{4}-\d{2}\b')
    uri_id_re = re.compile(r'\b\d{4}-\d{2}\b/.*')

    def __init__(
            self, language: str,
            snapshot_id: str,
            input_listings: List[str],
            input_base_uri: str,
            output_base_uri: str,
            log_dir: str,
            classifier_files: Dict[str, str],
            dsir_files: Dict[str, str],
            dsir_bucket: int,
            ldnoobw_dir: Path,
            ut1_dir: Path,
            minhash_similarities: List[float],
            minhash_ngram_size: int,
            minhash_num_permutations: int,
            monitor_queue: mp.Queue,
            logging_queue: mp.Queue,
            seed: int,
            endpoint_url: str = None,
            max_docs: int = -1,
            flush_interval=1000
    ):
        self._lang = language
        self._snapshot_id = snapshot_id
        self._input_base_uri = input_base_uri
        self._output_base_uri = output_base_uri
        self._dsir_files = dsir_files
        self._dsir_buckets = dsir_bucket
        self._flush_interval = flush_interval

        # init logger
        configure_worker_logger(logging_queue, level=logging.INFO)
        self._logger = logging.getLogger()

        # minhash setup
        self._minhash = MinHash(
            similarity_thresholds=minhash_similarities,
            ngram_size=minhash_ngram_size,
            num_permutations=minhash_num_permutations,
            seed=seed
        )

        self._logger.info(f"__MINHASH_PERM_CHECKSUM__ "
                          f"{self._minhash.checksum}")

        self._max_docs = max_docs
        self._monitor_queue = monitor_queue
        self._endpoint_url = endpoint_url

        self._job_id = str(uuid.uuid4())

        # build input paths
        self._input_uri_list = list(map(
            lambda x: os.path.join(self._input_base_uri, x),
            input_listings
        ))

        # init file to keep track of failed input files
        self._failed_input_file = os.path.join(
            log_dir, f"{language}-inputs.{self._job_id}.FAIL"
        )

        # init file to keep track of successful input files
        self._success_input_file = os.path.join(
            log_dir, f"{language}-inputs.{self._job_id}.SUCCESS"
        )

        # setup input file reader
        read_scheme = urlparse(self._input_base_uri).scheme
        if read_scheme == "s3":
            client = init_client(
                endpoint_url=self._endpoint_url,
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                signature_version="s3v4"
            )
        else:
            client = None

        self._reader = Reader(
            input_spec=InputSpec, threads=1, s3_client=client,
            logger=self._logger
        )

        # classifier model filepaths
        self._palm_model_file = classifier_files.get("palm")
        self._wikiref_model_file = classifier_files.get("wikiref")
        self._wikipedia_model_file = classifier_files.get("wikipedia")

        # initialize signal functions
        self._quality_signals = self.__init_quality_signals(
            ldnoobw_dir=ldnoobw_dir, ut1_dir=ut1_dir
        )

        # minhash_schema
        self._minhash_schema = pa.schema([
            ("shard_id", pa.string()),
            ("id", pa.string()),
            ("id_int", pa.uint64()),
            *[
                (
                    "signature_sim{s}".format(s=s), pa.list_(pa.binary())
                )
                for s in minhash_similarities
            ]
        ])

    @property
    def job_id(self):
        return self._job_id

    def __init_quality_signals(self, ldnoobw_dir, ut1_dir) -> List[Callable]:
        callables = []

        # initialize content signal functions
        self._logger.info(f"Registering content signals for {self._lang}..")
        callables += register_content_callables(
            language=self._lang,
            bad_urls_dir=ut1_dir,
            bad_words_dir=ldnoobw_dir
        )

        # initialize repetition removal signal functions
        self._logger.info(f"Registering repetition signals for {self._lang}..")
        callables += register_repetitions_callables()

        # initialize natural language signal functions
        self._logger.info(f"Registering natlang signals for {self._lang}..")
        callables += register_natural_language_callables()

        # initialize line signal functions
        self._logger.info(f"Registering line level signals for {self._lang}..")
        callables += register_lines_callables()

        # initialize ml heuristics signal functions
        self._logger.info(f"Registering classifier signals for {self._lang}..")
        callables += register_classifier_callables(
            wikiref_model=self._wikiref_model_file,
            palm_model=self._palm_model_file,
            wikipedia_model=self._wikipedia_model_file
        )

        # initialize importance weights signal functions
        # hacky -- first index is the counts file, second is the lambda file
        # this is set in pipeline.py
        self._logger.info(f"Registering dsir signals for {self._lang}..")
        callables += register_importance_weights_callables(
            source_fps=self._dsir_files.get("ccnet"),
            wiki_fps=self._dsir_files.get("wikipedia"),
            openwebtext_fps=self._dsir_files.get("openwebtext"),
            books_fps=self._dsir_files.get("books"),
            language=self._lang
        )

        return callables

    def __process_record(
            self, idx: int, record, uri_id: str, snapshot_id: str
    ):
        # Setup document; this precomputes ngrams and hash features
        document = Document(
            record.raw_content,
            domain=record.source_domain,
            precompute_ngrams=True,
            precompute_hash_features=True,
            dsir_buckets=self._dsir_buckets
        )

        # compute signals
        rp_v2_signals = {}
        for func in self._quality_signals:
            rp_v2_signals[func.field_name] = func(document)  # noqa

        # compute minhash signatures
        minhash_signatures = self._minhash.compute_banded_signatures(
            tokens=document.normalized_words
        )

        # compute document ids
        doc_id = f"{uri_id}/{idx}"
        doc_id_int = int.from_bytes(
            hashlib.sha1(doc_id.encode("utf-8")).digest()[:8],  # take 8 bytes
            byteorder=_BYTE_ORDER, signed=False
        )

        record_data = {
            "id": f"{uri_id}/{idx}",
            "id_int": doc_id_int,
        }

        metadata = {
            "cc_segment": record.cc_segment,
            "cc_net_source": uri_id,
            "url": record.url,
            "source_domain": record.source_domain,
            "language": record.language,
            "snapshot_id": snapshot_id
        }

        ccnet_quality_signals = {
            "ccnet_length": (
                (0, len(document), float(record.length)),
            ),
            "ccnet_original_length": (
                (0, len(document), float(record.original_length)),
            ),
            "ccnet_nlines": (
                (0, len(document), float(record.nlines)),
            ),
            "ccnet_original_nlines": (
                (0, len(document), float(record.original_nlines)),
            ),
            "ccnet_language_score": (
                (0, len(document), float(record.language_score)),
            ),
            "ccnet_perplexity": (
                (0, len(document), float(record.perplexity)),
            ),
            "ccnet_bucket": (
                (0, len(document), _ccnet_bucket_to_int(record.bucket)),
            ),
        }

        record_data["metadata"] = metadata
        record_data["quality_signals"] = {
            **ccnet_quality_signals, **rp_v2_signals
        }

        return record_data, minhash_signatures, doc_id, doc_id_int

    def __process_uri(self, docs_to_fetch: int, uri: str):
        num_docs = 0
        docs_added = 0
        snapshot_id = self.snapsh_re.search(uri).group(0)
        uri_id = self.uri_id_re.search(uri).group(0)

        # signal writer
        signal_uri = os.path.join(
            self._output_base_uri,
            self.shard_pattern_signals.format(shard_id=uri_id.split(".")[0]),
        )
        signal_writer = Writer(uri=signal_uri, schema=RP_SIGNAL_SCHEMA)
        self._logger.info(f"Initialized jsonl writer to {signal_uri}")

        # init minhash writer
        minhash_uri = os.path.join(
            self._output_base_uri,
            self.shard_pattern_minhash.format(shard_id=uri_id.split(".")[0]),
        )
        minhash_writer = ParquetBatchWriter(
            output_fp=minhash_uri, schema=self._minhash_schema
        )
        self._logger.info(f"Initialized parquet writer to {minhash_uri}")

        for idx, record in self._reader.read(
                uri=uri, max_samples=docs_to_fetch, return_idx=True
        ):
            # compute signals
            (
                record_data, minhash_signatures, doc_id, doc_id_int
            ) = self.__process_record(
                idx=idx, record=record, uri_id=uri_id, snapshot_id=snapshot_id
            )
            num_docs += 1
            docs_added += 1

            # write quality signals
            signal_writer.write(record_data)

            # record minhash signatures
            minhash_writer.update_batch(
                obj={"shard_id": uri_id, "id_int": doc_id_int, "id": doc_id,
                     **minhash_signatures}
            )

            # send to monitor
            if num_docs % self._flush_interval == 0:
                minhash_writer.write_batch()
                signal_writer.flush()
                self._monitor_queue.put({
                    "lang": self._lang, "num_docs": docs_added
                })
                docs_added = 0

        if docs_added > 0:
            self._monitor_queue.put({
                "lang": self._lang, "num_docs": docs_added
            })

        # close writers
        signal_writer.close()
        minhash_writer.close()

        gc.collect()

        return num_docs

    def run(self):
        total_docs = 0

        for i, uri in enumerate(self._input_uri_list, start=1):
            docs_to_fetch = self._max_docs - total_docs
            if docs_to_fetch <= 0 < self._max_docs:
                self._logger.info(
                    f"Reached max docs {self._max_docs} at {uri}")
                break

            # process file
            self._logger.info(
                f"Start processing {uri} ({i}/{len(self._input_uri_list)})"
            )
            try:
                docs_in_uri = self.__process_uri(docs_to_fetch, uri)
            except Exception as e:
                with open(self._failed_input_file, "a+") as f:
                    f.write(f"{uri}\n")
                self._logger.error(f"__URI_FAIL__ {uri} with exception: "
                                   f"{e.__class__.__name__}: {e} in "
                                   f"{self.__class__.__name__}.__process_uri")
                continue

            total_docs += docs_in_uri
            self._logger.info(
                f"__URI_SUCCESS__ {uri} ({i}/{len(self._input_uri_list)})"
            )

            # send signal that a uri has been completed
            self._monitor_queue.put({
                "lang": self._lang, "num_docs": None, "uri_complete": True
            })

            # keep track of completed uris
            with open(self._success_input_file, "a+") as f:
                f.write(f"{uri}\n")

        self._logger.info(f"Worker {self._job_id} Completed. "
                          f"Processed {total_docs} documents.")

        gc.collect()

        return total_docs, self._lang
