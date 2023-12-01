import argparse
import random

import boto3
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime as dt
import gzip
import io
import logging
import msgspec
import os
from pathlib import Path
import polars as pl
import progiter
import pyarrow as pa
import pybloomfilter
import re
from typing import Tuple
from urllib.parse import urlparse
from typing import Dict, List

from utilities.logging import configure_logger
from utilities.io import ParquetBatchWriter


@dataclass
class ReadStatus:
    is_success: bool
    msg: str
    uri: str


class Deduper:
    r""" Bloom filter for exact deduplication of ccnet shards. Based on
    document contents. """
    __slots__ = (
        "_args", "_logger", "_job_id", "_input_base_uri", "_scheme",
        "_output_fp", "_out_schema", "_bloom_fp"
    )

    # regex to extract filepaths from source file listings
    input_patterns = [
        re.compile(r".*/[a-z]{2}_middle\.json\.gz"),
        re.compile(r".*/[a-z]{2}_head\.json\.gz")
    ]

    output_pattern = "duplicates-{timestamp}-{snapshot}.parquet"

    def __init__(self):
        self._job_id = dt.now().strftime("%Y%m%d_%H%M%S")
        self._args = self.__parse_arguments()

        # set random seed
        random.seed(self._args.seed)

        # parse args
        self._input_base_uri = self._args.input_base_uri
        self._scheme = urlparse(self._input_base_uri).scheme

        # init logging
        logfile = Path(self._args.output_dir) / "logs" / f"{self._job_id}.log"
        configure_logger(logfile=logfile, level=logging.INFO, stream=False)
        self._logger = logging.getLogger()

        # output writer
        self._output_fp = Path(self._args.output_dir) / "duplicates.parquet"
        self._out_schema = pa.schema([
            ("shard_id", pa.string()),
            ("doc_id", pa.string()),
            ("digest", pa.string())
        ])

        # log setup
        for attr in [
            "listings", "input_base_uri", "output_dir", "parallel_readers",
            "capacity", "error_rate", "seed", "max_inputs", "batch_size"
        ]:
            self._logger.info(f"{attr}: {getattr(self._args, attr)}")

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
            "--listings", type=str, default=None,
            help="Path to a file containing paths to ccnet shards; needs to "
                 "match with the input_base_uri argument."
        )
        parser.add_argument(
            "--input_base_uri", type=str, default=None,
            help="base uri of the input files."
        )
        parser.add_argument(
            "--output_dir", type=str, default=None,
            help="directory where the output will be stored."
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
            "--parallel_readers", type=int, default=1,
            help="number of parallel reader processes. Defaults to 1."
        )
        parser.add_argument(
            "--max_inputs", type=int, default=None,
            help="maximum number of inputs to process. For debugging."
        )
        parser.add_argument(
            "--batch_size", type=int, default=None,
            help="number of listings to be processed per process."
        )

        parser.add_argument(
            "--seed", type=int, default=42,
            help="random seed."
        )

        # dedup params
        parser.add_argument(
            "--capacity", type=int, default=None,
            help="Capacity of the bloom filter. This is the maximum number of "
                 "unique documents that can be stored in the filter while "
                 "keeping the error rate under `error_rate`."
        )
        parser.add_argument(
            "--error_rate", type=float, default=0.01,
            help="false positive probability that will hold given that "
                 "'capacity' is not exceeded. Defaults to 0.001"
        )
        args = parser.parse_args()

        return args

    def __init_client(self):
        if self._scheme != "s3":
            return None

        session = boto3.Session(profile_name=self._args.s3_profile)
        return session.client(
            service_name='s3',
            endpoint_url=self._args.endpoint_url,
            config=boto3.session.Config(
                signature_version="s3v4",
                retries={'max_attempts': 5, 'mode': 'standard'}
            )
        )

    def __filter_listings(self, obj_key: str):
        for pat in self.input_patterns:
            if pat.search(obj_key) is not None:
                return True

        return False

    def __parse_listings(self):
        # build input uris
        with open(self._args.listings, "r") as f:
            uris = list(
                map(lambda ls: os.path.join(self._input_base_uri, ls.strip()),
                    filter(self.__filter_listings, f.readlines()))
            )

        return uris

    @staticmethod
    def __load_from_s3(uri, client):
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

        return is_success, msg, buffer

    @staticmethod
    def __load_from_disk(uri):
        try:
            with open(uri.path, "rb") as f:
                buffer = io.BytesIO(f.read())
                msg = f"__DISK_URI_READ_SUCCESS__ success reading {uri.path}"
                is_success = True
        except Exception as e:
            msg = (
                f"__DISK_URI_READ_ERROR__ failed reading {uri.path}: "
                f"caught exception {e.__class__.__name__}: {e}"
            )
            buffer = None
            is_success = False

        return is_success, msg, buffer

    def _load_file(self, uri, client) -> Tuple[ReadStatus, io.BytesIO]:
        if uri.scheme == "s3":
            is_success, msg, buffer = self.__load_from_s3(uri, client)
        elif uri.scheme == "file":
            is_success, msg, buffer = self.__load_from_disk(uri)
        else:
            raise ValueError(f"Unknown scheme {uri.scheme}")

        read_status = ReadStatus(
            is_success=is_success, msg=msg, uri=uri.geturl()
        )
        return read_status, buffer

    def _load_and_parse_inputs(
            self, input_chunk
    ) -> Dict[str, Tuple[ReadStatus, List[Dict]]]:
        # build msgspec decoder
        decoder = msgspec.json.Decoder(
            type=msgspec.defstruct(name="Record", fields=[("digest", str)])
        )

        client = self.__init_client()
        data = {}

        for uri in input_chunk:
            read_status, buffer = self._load_file(
                uri=urlparse(uri), client=client
            )

            if not read_status.is_success:
                data[uri] = (read_status, [])
                continue

            shard_id = read_status.uri.replace(
                self._input_base_uri, ""
            ).lstrip("/")

            uri_data = []

            try:
                with gzip.open(buffer, "rb") as f:
                    for idx, obj in enumerate(f):
                        rec = decoder.decode(obj)
                        digest = str(getattr(rec, "digest")).replace(
                            "sha1:", ""
                        )
                        uri_data.append({
                            "shard_id": shard_id,
                            "doc_id": f"{shard_id}/{idx}",
                            "digest": digest
                        })
            except Exception as e:
                uri_data = []
                read_status.msg = (
                    f"__S3_URI_DECODE_ERROR__ failed decoding {uri}: "
                    f"caught exception {e.__class__.__name__}: {e}"
                )
                read_status.is_success = False

            data[uri] = (read_status, uri_data)

        del buffer

        return data

    def __parallel_run(self, input_uris):
        # shuffle input uris
        random.shuffle(input_uris)

        if self._args.max_inputs is not None:
            self._logger.info(f"Limiting inputs to {self._args.max_inputs}")
            input_uris = input_uris[:self._args.max_inputs]

        # divide input uris into snapshots
        snapsh_re = re.compile(r'\b\d{4}-\d{2}\b')
        snapshots = {}
        for uri in input_uris:
            snapshot = snapsh_re.search(uri).group(0)
            if snapshot not in snapshots:
                snapshots[snapshot] = [uri]
            else:
                snapshots[snapshot].append(uri)

        snapshot_ids_sorted = sorted(snapshots.keys(), reverse=True)

        # init bloomfilter
        bloomfilter = pybloomfilter.BloomFilter(
            capacity=self._args.capacity,
            error_rate=self._args.error_rate
        )

        self._logger.info(f"Filter capacity: {bloomfilter.capacity}")
        self._logger.info(f"Filter error rate: {bloomfilter.error_rate}")
        self._logger.info(f"Filter hash seeds: {bloomfilter.hash_seeds}")

        num_docs, num_dupes = 0, 0

        # progress bars
        pman = progiter.ProgressManager(backend="rich")
        total_progress = pman.progiter(
            total=self._args.capacity, postfix_str="Duplicates: --"
        )
        download_progress = pman.progiter(
            total=len(input_uris), desc="Download"
        )

        num_failed_uri = 0
        num_succ_uri = 0

        for snapsh_id in snapshot_ids_sorted:
            uri_list = snapshots[snapsh_id]
            random.shuffle(uri_list)

            uri_list_partitioned = [
                uri_list[i:i + self._args.batch_size]
                for i in range(0, len(uri_list), self._args.batch_size)
            ]

            self._logger.info(f"__SNAPSHOT_START__ {snapsh_id}")

            # output writer
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            output_fp = (
                    Path(self._args.output_dir) /
                    self.output_pattern.format(
                        timestamp=timestamp, snapshot=snapsh_id
                    )
            )
            out_writer = ParquetBatchWriter(
                output_fp=output_fp, schema=self._out_schema
            )

            try:
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=self._args.parallel_readers
                ) as executor:
                    futures = {
                        executor.submit(
                            self._load_and_parse_inputs, input_chunk
                        ): i
                        for i, input_chunk in enumerate(uri_list_partitioned)
                    }

                    for future in concurrent.futures.as_completed(futures):
                        data_chunks = future.result()
                        del futures[future]
                        download_progress.step(len(data_chunks))

                        for (
                                uri, (read_status, uri_data)
                        ) in data_chunks.items():

                            if not read_status.is_success:
                                self._logger.error(read_status.msg)
                                num_failed_uri += 1
                                continue

                            num_succ_uri += 1
                            download_progress.set_postfix_str(
                                f"success: {num_succ_uri} "
                                f"({num_failed_uri} failed)"
                            )

                            self._logger.info(read_status.msg)

                            for record in uri_data:
                                digest = record["digest"]

                                if bloomfilter.add(digest):
                                    out_writer.update_batch(obj=record)
                                    num_dupes += 1

                                num_docs += 1
                                total_progress.step(1)

                                if num_docs % (1024 ** 2) == 0:
                                    out_writer.write_batch()

                            dupe_prop = round(100 * num_dupes / num_docs, 2)
                            total_progress.set_postfix_str(
                                f"Duplicates: {num_dupes} / {num_docs}"
                                f" ({dupe_prop:.2f}%)"
                            )

            except KeyboardInterrupt:
                self._logger.info("Keyboard interrupt. Stopping.")
                executor.shutdown(wait=False, cancel_futures=True)
                out_writer.close()
                break
            except Exception as e:
                self._logger.error(
                    f"Caught exception {e.__class__.__name__}: {e}"
                )
                executor.shutdown(wait=False, cancel_futures=True)
                out_writer.close()
                self._logger.info(f"__SNAPSHOT_FAIL__ {snapsh_id}")
                continue

            out_writer.close()
            self._logger.info(f"__SNAPSHOT_FINISH__ {snapsh_id}")

        pman.stop()
        bloomfilter.close()

        self._logger.info(f"Filtering complete.")

    def run(self):
        start_time = dt.now()
        print(f"start @ {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.__parallel_run(input_uris=self.__parse_listings())
        end_time = dt.now()
        print(f"end @ {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        end_str = f"Total time: {end_time - start_time}"
        print(end_str)
        self._logger.info(end_str)

    def __result_summary(self):
        dump_reg = "(\d{4}-\d{2})\/"
        # read duplicates
        query = (
            pl.scan_parquet(self._output_fp)
            .with_columns(
                pl.col("shard_id").str.extract(dump_reg, 1).alias("snapshot")
            )
            .group_by("snapshot")
            .agg(pl.count())
        )

        stats = query.collect()

        with pl.Config(fmt_str_lengths=1000, tbl_rows=100):
            print(stats)


if __name__ == '__main__':
    deduper = Deduper()
    deduper.run()
