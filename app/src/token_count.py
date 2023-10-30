import argparse
import boto3
import botocore.client
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime as dt
import gzip
import io
import logging
import msgspec
import os
from pathlib import Path
import progiter
import pyarrow as pa
import random
import re
from typing import Tuple, List
from tokenizers import Tokenizer
from urllib.parse import urlparse, ParseResult

from utilities.logging import configure_logger
from utilities.io.writer import ParquetBatchWriter


class InputSpec(msgspec.Struct):
    raw_content: str


@dataclass
class DlStatus:
    is_success: bool
    msg: str
    uri: str


@dataclass
class InputResult:
    is_success: bool
    msg: str
    input_id: str
    num_docs: int = 0
    num_tokens: int = 0
    token_counts: List[Tuple[int, int]] = None


TOKENIZER = Tokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


class PostProcessor:
    listings_re = re.compile(
        r".*(\d{4}-\d{2}/\d{4}/(?:en|es|de|fr|it)_(?:tail|middle|head)).json.gz"
    )

    def __parse_arguments(self) -> argparse.Namespace:

        if self.__doc__ is not None:
            description = " - " + self.__doc__
        else:
            description = self.__class__.__name__

        parser = argparse.ArgumentParser(
            prog=self.__class__.__name__, description=description
        )

        # io
        parser.add_argument(
            "--snapshots", nargs="+", type=str, default=None,
        )
        parser.add_argument(
            "--input_base_uri", type=str, default=None,
            help="base uri of the input files."
        )
        parser.add_argument(
            "--logs_dir", type=str, default=None,
            help="directory to store logs."
        )

        parser.add_argument(
            "--s3_profile", type=str, default="default",
            help="profile name of the s3 client."
        )
        parser.add_argument(
            "--endpoint_url", type=str, default=None,
            help="S3 bucket endpoint url."
        )
        parser.add_argument(
            "--parallelism", type=int, default=1,
            help="number of parallel processes. Defaults to 1."
        )

        parser.add_argument(
            "--batch_size", type=int, default=1,
            help="batch size. Defaults to 1."
        )
        parser.add_argument(
            "--max_inputs", type=int, default=4,
            help="maximum number of inputs to process. For debugging."
        )

        parser.add_argument(
            "--debug", default=0, choices=[0, 1], type=int,
            help="runs in debug mode if set to 1."
        )
        parser.add_argument(
            "--input_listings", type=str, default="listings.txt",
            help="path to file containing input ids."
        )
        parser.add_argument(
            "--seed", type=int, default=42
        )

        args = parser.parse_args()

        return args

    def __init__(self):
        self._job_id = dt.now().strftime("%Y%m%d_%H%M%S")
        self._args = self.__parse_arguments()

        random.seed(self._args.seed)

        # i/o
        self._input_base_uri = self._args.input_base_uri
        self._logs_dir = self._args.logs_dir

    def __init_client(self):
        session = boto3.Session(profile_name=self._args.s3_profile)
        client = session.client(
            service_name='s3',
            endpoint_url=self._args.endpoint_url,
            config=boto3.session.Config(
                signature_version='s3v4',
                retries={'max_attempts': 10, 'mode': 'standard'}
            )
        )
        return session, client

    @staticmethod
    def _dload_file(uri: ParseResult, client) -> Tuple[DlStatus, io.BytesIO]:
        try:
            streaming_body = client.get_object(
                Bucket=uri.netloc, Key=uri.path.lstrip("/")
            )["Body"]
            buffer = io.BytesIO(streaming_body.read())
            msg = f"__S3_URI_READ_SUCCESS__ success reading {uri.path}"
            is_success = True
        except Exception as e:
            msg = (
                f"__S3_URI_READ_ERROR__ failed reading {uri.path}: "
                f"caught exception {e.__class__.__name__}: {e}"
            )
            buffer = None
            is_success = False

        read_status = DlStatus(is_success=is_success, msg=msg, uri=str(uri))
        return read_status, buffer

    def __load_input_ids(
            self, snapshot: str
    ) -> List[str]:

        assert self._args.input_listings is not None

        input_ids = []
        with open(self._args.input_listings, "r") as fin:
            for ln in fin.readlines():
                try:
                    ln = self.listings_re.findall(ln.strip())[0]
                except IndexError:
                    continue
                if f"{snapshot}/" not in ln:
                    continue
                input_ids.append(ln)

        return input_ids

    def _process_listings(self, input_ids: List[str]) -> List[InputResult]:
        sess, client = self.__init_client()

        # decoding and encoding
        decoder = msgspec.json.Decoder(type=InputSpec)

        results = []
        for input_id in input_ids:
            proc_res: InputResult = self._process_single_listing(
                client, input_id, decoder
            )
            results.append(proc_res)

        return results

    def _process_single_listing(
            self, client, input_id, decoder
    ) -> InputResult:
        # handle signals
        result: InputResult = self._handle_documents(
            client, input_id, decoder
        )
        if not result.is_success:
            result.msg = f"__FAIL__ {input_id} ({result.msg})"
            return result

        result.msg = f"__SUCCESS__ {input_id}"

        return result

    def _handle_documents(
            self,
            client: botocore.client.BaseClient,
            input_id: str,
            decoder
    ) -> InputResult:
        # download doc
        input_uri = urlparse(
            os.path.join(
                self._input_base_uri, f"{input_id}.json.gz"
            )
        )
        dl_status, input_buffer = self._dload_file(input_uri, client=client)

        # check if download was successful
        if not dl_status.is_success:
            return InputResult(
                is_success=False, msg=dl_status.msg, input_id=input_id
            )

        num_docs = 0
        total_tokens = 0
        token_counts = []

        try:
            with gzip.open(input_buffer, mode="rb") as in_fh:
                for idx, obj in enumerate(in_fh):
                    record = decoder.decode(obj)

                    # tokenize
                    num_tokens = len(
                        TOKENIZER.encode(record.raw_content).tokens
                    )
                    token_counts.append((idx, num_tokens))

                    total_tokens += num_tokens
                    num_docs += 1

        except Exception as e:
            msg = (
                f"__DECODE_ENCODE_FAIL__ {input_id}: "
                f"caught exception {e.__class__.__name__}: {e}"
            )
            return InputResult(is_success=False, msg=msg, input_id=input_id)

        return InputResult(
            is_success=True,
            msg="",
            input_id=input_id,
            num_docs=num_docs,
            num_tokens=total_tokens,
            token_counts=token_counts
        )

    def run(self):
        # init logging
        logfile = Path(self._logs_dir) / f"{self._job_id}.log"
        configure_logger(logfile=logfile, level=logging.INFO, stream=False)
        logger = logging.getLogger()

        # log configs
        for attr in (
                "snapshots", "input_base_uri", "batch_size",
                "parallelism", "max_inputs", "debug", "input_listings", "seed"
        ):
            logger.info(f"__CONFIG__ {attr}: {getattr(self._args, attr)}")

        for snapshot in self._args.snapshots:
            logger.info(f"__START_SNAPSHOT__ {snapshot}")
            try:
                self.run_snapshot(snapshot, logger=logger)
            except KeyboardInterrupt:
                break
            logger.info(f"__END_SNAPSHOT__ {snapshot}")

    def run_snapshot(self, snapshot_id, logger):
        # load input file ids
        input_ids = self.__load_input_ids(snapshot_id)
        msg = (
            f"__INPUT_LISTINGS_LOADED__ "
            f"found {len(input_ids)} input files in {snapshot_id}"
        )
        logger.info(msg)
        random.shuffle(input_ids)

        if self._args.max_inputs is not None:
            input_ids = input_ids[:self._args.max_inputs]

        input_ids_batches = [
            input_ids[i:i + self._args.batch_size]
            for i in range(0, len(input_ids), self._args.batch_size)
        ]

        # init output writer
        out_fp = Path(self._logs_dir) / f"{snapshot_id}_counts.parquet"
        out_schema = pa.schema([
            ("input_id", pa.string()),
            ("doc_id", pa.string()),
            ("snapshot_id", pa.string()),
            ("num_tokens", pa.int64())
        ])

        pq_writer = ParquetBatchWriter(output_fp=out_fp, schema=out_schema)

        if self._args.debug:
            self.__debug_run(
                input_ids_batches, logger=logger, snapshot_id=snapshot_id,
                pq_writer=pq_writer
            )
        else:
            self.__parallel_run(
                input_ids_batches, logger=logger, snapshot_id=snapshot_id,
                pq_writer=pq_writer
            )

        pq_writer.close()

    def __debug_run(
            self,
            input_ids_batches: List[List[str]],
            logger: logging.Logger,
            snapshot_id: str,
            pq_writer: ParquetBatchWriter
    ):
        num_docs = 0
        num_succ = 0
        num_fail = 0
        total_tokens = 0

        # progress bar
        total_inputs = sum(map(len, input_ids_batches))
        pman = progiter.ProgressManager(backend="rich")
        pbar = pman.progiter(
            total=total_inputs,
            desc=f"Processing {snapshot_id}",
            backend="rich"
        )

        for batch in input_ids_batches:
            inputs_results: List[InputResult] = self._process_listings(batch)

            for proc_res in inputs_results:
                if proc_res.is_success:
                    num_succ += 1
                    num_docs += proc_res.num_docs
                    total_tokens += proc_res.num_tokens
                else:
                    num_fail += 1

                logger.info(proc_res.msg)

                pbar.step(1)
                pbar.set_postfix_str(
                    f"total_inputs: {num_succ:,} ({num_fail:,} fail); "
                    f"num_docs: {num_docs:,} -- "
                    f"num_tokens: {total_tokens:,}"
                )

                if not proc_res.is_success:
                    continue

                for idx, num_tokens in proc_res.token_counts:
                    pq_writer.update_batch({
                        "input_id": proc_res.input_id,
                        "doc_id": f"{proc_res.input_id}.json.gz/{idx}",
                        "snapshot_id": snapshot_id,
                        "num_tokens": num_tokens,
                    })

            pq_writer.write_batch()

        pman.stop()

        # log summary
        logger.info(
            f"__PROCESSING_COMPLETE__\n*******************\n"
            f"num_inputs_success: {num_succ:,}\n"
            f"num_inputs_failed: {num_fail:,}\n"
            f"num_docs: {num_docs:,}\n"
            f"num_tokens: {total_tokens:,}"
        )

    def __parallel_run(
            self,
            input_ids_batches: List[List[str]],
            logger: logging.Logger,
            snapshot_id: str,
            pq_writer: ParquetBatchWriter
    ):
        num_docs = 0
        num_succ = 0
        num_fail = 0
        total_tokens = 0

        # progress bar
        total_inputs = sum(map(len, input_ids_batches))
        pman = progiter.ProgressManager(backend="rich")
        pbar = pman.progiter(
            total=total_inputs,
            desc=f"Processing {snapshot_id}",
            backend="rich"
        )

        # process listings
        try:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self._args.parallelism
            ) as executor:
                futures = {
                    executor.submit(
                        self._process_listings,
                        input_ids=batch,
                    ): batch
                    for batch in input_ids_batches
                }

                for future in concurrent.futures.as_completed(futures):
                    proc_results: List[InputResult] = future.result()
                    del futures[future]

                    for proc_res in proc_results:
                        if proc_res.is_success:
                            num_succ += 1
                            num_docs += proc_res.num_docs
                            total_tokens += proc_res.num_tokens
                        else:
                            num_fail += 1

                        logger.info(proc_res.msg)

                        pbar.step(1)
                        pbar.set_postfix_str(
                            f"total_inputs: {num_succ:,} ({num_fail:,} fail); "
                            f"num_docs: {num_docs:,} -- "
                            f"num_tokens: {total_tokens:,}"
                        )

                        if not proc_res.is_success:
                            continue

                        for idx, num_tokens in proc_res.token_counts:
                            pq_writer.update_batch({
                                "input_id": proc_res.input_id,
                                "doc_id": f"{proc_res.input_id}.json.gz/{idx}",
                                "snapshot_id": snapshot_id,
                                "num_tokens": num_tokens,
                            })

                    pq_writer.write_batch()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught. Terminating...")
            pman.stop()
            executor.shutdown(wait=False, cancel_futures=True)
            pq_writer.close()
            raise KeyboardInterrupt

        pman.stop()

        # log summary
        logger.info(
            f"__PROCESSING_COMPLETE__\n*******************\n"
            f"num_inputs_success: {num_succ:,}\n"
            f"num_inputs_failed: {num_fail:,}\n"
            f"num_docs: {num_docs:,}\n"
            f"num_tokens: {total_tokens:,}"
        )


if __name__ == '__main__':
    pp = PostProcessor()
    pp.run()
