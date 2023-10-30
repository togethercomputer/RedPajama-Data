from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

from utilities.io import Writer


class OpenWebTextDownloader:
    r""" Loads the Openwebtext dataset from HuggingFace Datasets and saves it
    to disk """

    dataset_name = "openwebtext"
    output_fp = "openwebtext/en-openwebtext.jsonl.gz"

    def __init__(self, lang, out_dir, overwrite, cache_dir, max_samples):
        self._lang = lang
        self._out_dir = out_dir
        self._overwrite = overwrite
        self._cache_dir = cache_dir
        self._max_samples = max_samples
        self._filepath = None

    def __str__(self):
        return f"{self.__class__.__name__}(lang={self._lang})"

    @property
    def filepath(self):
        return self._filepath

    def run(self, logger):
        if self._lang != "en":
            logger.info(f"{str(self)} Skipping {self._lang}")
            return

        self._filepath = Path(self._out_dir) / self.output_fp
        logger.info(f"{str(self)} Output file: {self._filepath}")
        logger.info(f"{str(self)} max_samples: {self._max_samples}")

        if self._filepath.exists():
            if not self._overwrite:
                raise FileExistsError(f"File {self._filepath} already exists.")
            else:
                self._filepath.unlink()
                logger.info(f"{str(self)} Deleted {self._filepath}")

        out_uri = "file://" + str(self._filepath)
        writer = Writer(uri=out_uri, schema=[("text", str)])

        logger.info(f"{str(self)} Download start.")
        pbar = tqdm(desc="writing progress")
        flush_every = 10_000

        n_docs = 0
        for record in load_dataset(
                "openwebtext", cache_dir=self._cache_dir, split="train",
                streaming=True
        ):
            n_docs += 1
            if n_docs > self._max_samples > 0:
                break
            writer.write(
                data_obj={"text": record["text"]},
                flush=n_docs % flush_every == 0
            )
            pbar.update(1)

        pbar.close()
        writer.close()
        logger.info(f"{str(self)} Download finished; num_samples={n_docs - 1}")
