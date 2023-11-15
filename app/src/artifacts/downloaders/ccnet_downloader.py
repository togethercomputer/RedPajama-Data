from collections import defaultdict
import subprocess
from pathlib import Path
import re
import functools
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.pool import Pool
from urllib.parse import urlparse
import os

from utilities.io import Reader, Writer
from utilities.io.s3 import init_client


class CCNetDownloader(object):
    r"""
    This class downloads / loads ccnet data and writes it to a jsonl file.
    """

    dataset_name = "ccnet"

    # extension of the cc input files
    cc_ext = ".json.gz"

    def __init__(
            self,
            artifacts_dir,
            cc_input,
            cc_input_base_uri,
            lang, num_samples, max_workers,
            endpoint_url
    ):
        # write args to class variables
        self._lang = lang
        self._num_samples = num_samples
        self._cc_input = cc_input
        self._cc_input_base_uri = cc_input_base_uri
        self._endpoint_url = endpoint_url

        # parallel readers
        if max_workers is not None:
            self._parallel_readers = max_workers
        else:
            self._parallel_readers = mp.cpu_count() - 2

        # build output path
        self._output_fp = Path(artifacts_dir) / "datasets" / \
                          self._lang / "ccnet" / "ccnet.jsonl"
        self._output_fp.parent.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return f"{self.__class__.__name__}({self._lang})"

    @property
    def filepath(self):
        return self._output_fp

    def __ccnet_file_filter(self, fp: str) -> bool:
        r""" function to filter commoncrawl  input files. """
        # we only keep files in the target language
        if not Path(fp).name.startswith(f"{self._lang}_"):
            return False

        # check extension
        if not fp.endswith(self.cc_ext):
            return False

        return True

    def run(self, logger):
        if not Path(self._cc_input).exists():
            raise ValueError(
                f"Listings file {self._cc_input} does not exist"
            )

        # read the listings file and return the relative paths listed in the
        # file.
        logger.info(f"{str(self)} Start loading input listings...")
        with open(self._cc_input) as f:
            input_listings = list(map(
                lambda _fp: os.path.join(self._cc_input_base_uri, _fp),
                filter(self.__ccnet_file_filter, map(str.strip, f.readlines()))
            ))

        # partition cc input by snapshot id in order to ensure that we have a
        # balanced number of samples per snapshot. This is to avoid bias due
        # to distribution shifts over time.
        logger.info(f"{str(self)} Partitioning inputs by snapshot...")
        snapsh_re = re.compile(r'\b\d{4}-\d{2}\b')
        inputs_by_snapsh = defaultdict(list)
        for listing in input_listings:
            if (dump_id := snapsh_re.search(listing).group()) is None:
                continue
            inputs_by_snapsh[dump_id].append(listing)

        samples_per_snapshot = max(
            1, self._num_samples // len(inputs_by_snapsh)
        )

        # kick off processes
        manager = mp.Manager()
        data_queue = manager.Queue(maxsize=128 * self._parallel_readers)

        # writer
        writer_proc = mp.Process(
            target=self._writer_worker, args=(data_queue,)
        )
        writer_proc.start()

        logger.info(f"{str(self)} Start loading {self._num_samples} samples "
                    f"from {len(inputs_by_snapsh)} snapshots")

        with Pool(processes=self._parallel_readers) as pool:
            counts_per_snapsh = pool.starmap(
                functools.partial(self._load_snapshot, data_queue=data_queue),
                [
                    (snpsh_id, snpsh_files, samples_per_snapshot)
                    for snpsh_id, snpsh_files in inputs_by_snapsh.items()
                ]
            )

        total_samples = 0
        for counts, snapshot_id in counts_per_snapsh:
            logger.info(f"{str(self)} Snapshot {snapshot_id}: "
                        f"loaded {counts} samples.")
            total_samples += counts

        logger.info(f"{str(self)} Total: loaded {total_samples} samples.")
        logger.info(f"{str(self)} Shuffling...")
        subprocess.run(["shuf", self._output_fp, "-o", self._output_fp])
        logger.info(f"{str(self)} Done. Output: {self._output_fp}")

        # send kill signal to writer
        data_queue.put_nowait(None)
        writer_proc.join()
        manager.shutdown()

    def _load_snapshot(
            self, snapshot_id, input_uris, num_samples, data_queue: mp.Queue,
    ):
        # partition input files into head, middle and tail
        head_uris = list(filter(lambda _u: "_head" in _u, input_uris))
        middle_uris = list(filter(lambda _u: "_middle" in _u, input_uris))
        tail_uris = list(filter(lambda _u: "_tail" in _u, input_uris))

        # compute number of samples to load from each bucket
        samples_per_bucket = {
            "head": int(num_samples * 0.1),
            "middle": int(num_samples * 0.2),
            "tail": int(num_samples * 0.7)
        }

        if urlparse(self._cc_input_base_uri).scheme == "s3":
            s3_client = init_client(
                endpoint_url=self._endpoint_url,
                signature_version="s3v4",
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
            )
        else:
            s3_client = None

        reader = Reader(
            schema=[("raw_content", str), ("language", str)],
            s3_client=s3_client
        )

        total_samples = 0

        for bucket, bucket_list in zip(
                ["head", "middle", "tail"], [head_uris, middle_uris, tail_uris]
        ):
            samples_retrieved = 0
            target_samples = samples_per_bucket[bucket]

            for uri in bucket_list:

                samples_to_retrieve = target_samples - samples_retrieved
                if samples_to_retrieve <= 0:
                    break

                for idx, record in reader.read(
                        uri=uri, max_samples=samples_to_retrieve
                ):
                    data_queue.put({
                        "text": record.raw_content,
                        "lang": record.language,
                        "source": uri
                    })

                    samples_retrieved += 1
                    total_samples += 1

        return total_samples, snapshot_id

    def _writer_worker(self, data_queue: mp.Queue):

        writer = Writer(
            uri="file://" + str(self._output_fp),
            schema=[("text", str), ("lang", str), ("source", str)]
        )

        flush_every = 10_000

        pbar = tqdm(desc="writing progress")

        num_recs = 0
        while True:
            data = data_queue.get()

            if data is None:
                break

            num_recs += 1
            writer.write(data, flush=num_recs % flush_every == 0)
            pbar.update(1)

        pbar.close()
        writer.close()
