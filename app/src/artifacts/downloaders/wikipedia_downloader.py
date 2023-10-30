from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

from utilities.io import Writer


class WikipediaDownloader:
    r""" Loads the Wikipedia dataset from HuggingFace Datasets and saves it to
    disk """

    dataset_name = "wikipedia"
    output_pattern = "wikipedia/{lang}-wikipedia.jsonl.gz"

    def __init__(self, lang, out_dir, overwrite, cache_dir, max_samples):
        self._lang = lang
        self._out_dir = out_dir
        self._overwrite = overwrite
        self._cache_dir = cache_dir
        self._max_samples = max_samples
        self._filepath = None

    def __str__(self):
        return f"{self.__class__.__name__}({self._lang})"

    @property
    def filepath(self):
        return self._filepath

    def run(self, logger):
        output_fn = self.output_pattern.format(lang=self._lang)
        self._filepath = Path(self._out_dir) / output_fn

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

        try:
            # try to load wikipedia data from preprocessed huggingface dataset
            ds_iterator = load_dataset(
                "wikipedia", f"20220301.{self._lang}", streaming=True,
                split="train"
            )
            logger.info(f"{str(self)} Load {self._lang}-wiki from 20220301")
        except Exception as _:
            # if that fails, load from original huggingface dataset and process
            ds_iterator = load_dataset(
                "wikipedia", language=self._lang, date="20230801",
                cache_dir=self._cache_dir, beam_runner="DirectRunner",
                split="train"
            )
            logger.info(f"{str(self)} Load {self._lang}-wiki from 20230801")

        n_docs = 0
        for record in ds_iterator:
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
