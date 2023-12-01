import argparse
import logging
from datetime import datetime as dt
import os
from pathlib import Path

from artifacts.downloaders import (
    WikipediaDownloader,
    OpenWebTextDownloader,
    BooksDownloader,
    CCNetDownloader
)
from artifacts.hash_dist import HashDist
from artifacts.ft_trainer import FastTextTrainer
from utilities.logging.configure import configure_logger


def parse_arguments():
    def nullable_string(val):
        # converts empty string to None
        return None if not val else val

    parser = argparse.ArgumentParser()
    # input and outputs
    parser.add_argument(
        "--artifacts_dir", type=str, default=None,
        help="Directory where artifacts of the pipeline are stored"
    )
    parser.add_argument(
        "--cc_input", type=str, default=None,
        help="cc_net output listings"
    )
    parser.add_argument(
        "--cc_input_base_uri", type=str, default=None,
        help="Base URL (prefix) used for files list in input. Used to "
             "select the access method: s3://<path>/ or file://<path>/"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="huggingface cache directory"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "--lang", type=str, default=None
    )
    parser.add_argument(
        "--max_workers", type=int, default=None,
        help="Maximum number of workers to use"
    )
    parser.add_argument(
        "--dsir_num_samples", type=int, default=None,
        help="Number of samples to use for dsir"
    )
    parser.add_argument(
        "--dsir_feature_dim", type=int, default=None,
        help="Number of buckets to use for dsir"
    )
    parser.add_argument(
        "--classifiers_num_samples", type=int, default=None,
        help="Number of samples to use for classifiers"
    )
    parser.add_argument(
        "--endpoint_url", type=nullable_string, default=None,
        help="endpoint url where the s3 bucket is exposed."
    )

    # sampling
    parser.add_argument(
        "--max_samples_per_book", type=int, default=None,
        help="Maximum number of samples to use per book"
    )
    parser.add_argument(
        "--max_paragraphs_per_book_sample", type=int, default=None,
        help="Maximum number of paragraphs to use per book sample"
    )

    return parser.parse_args()


def main(artifacts_dir: str, cc_input: str, cc_input_base_uri: str,
         cache_dir: str, overwrite: bool, lang: str,
         max_workers: int, endpoint_url: str,
         dsir_num_samples: int, dsir_feature_dim: int,
         classifiers_num_samples: int, max_samples_per_book: int,
         max_paragraphs_per_book_sample: int
         ):
    if max_workers is None:
        max_workers = os.cpu_count() - 2
    else:
        max_workers = min(max_workers, os.cpu_count() - 2)

    # parse config
    num_samples = max(dsir_num_samples, classifiers_num_samples)

    # build output directory
    datasets_dir = Path(artifacts_dir) / "datasets" / f"{lang}"
    datasets_dir.mkdir(exist_ok=True, parents=True)
    timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
    logfile = Path(artifacts_dir) / f"logs/{lang}_artifacts@{timestamp}.log"
    logfile.parent.mkdir(exist_ok=True, parents=True)
    configure_logger(logfile=logfile, level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Start preparing artifacts for {lang}")
    logger.info(f"num_samples: {num_samples}")
    logger.info(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")

    # download ccnet dataset
    ccnet = CCNetDownloader(
        lang=lang, artifacts_dir=artifacts_dir, cc_input=cc_input,
        cc_input_base_uri=cc_input_base_uri, num_samples=num_samples,
        max_workers=max_workers, endpoint_url=endpoint_url
    )
    ccnet.run(logger=logger)

    # download wikipedia dataset
    wikipedia = WikipediaDownloader(
        lang=lang, out_dir=datasets_dir,
        overwrite=overwrite, cache_dir=cache_dir,
        max_samples=num_samples
    )
    wikipedia.run(logger=logger)

    # download openwebtext dataset
    openwebtext = OpenWebTextDownloader(
        lang=lang, out_dir=datasets_dir,
        overwrite=overwrite, cache_dir=cache_dir,
        max_samples=num_samples
    )
    openwebtext.run(logger=logger)

    # download books dataset
    books = BooksDownloader(
        lang=lang, out_dir=datasets_dir,
        overwrite=overwrite, cache_dir=cache_dir,
        max_samples=num_samples,
        max_paragraphs_per_sample=max_paragraphs_per_book_sample,
        max_samples_per_book=max_samples_per_book,
    )
    books.run(logger=logger)

    # compute hash distributions
    hash_dist = HashDist(
        artifacts_dir=artifacts_dir,
        num_samples=num_samples,
        buckets=dsir_feature_dim,
        max_workers=max_workers,
        logger=logger
    )

    # compute hash distribution for each dataset
    for obj in [wikipedia, openwebtext, books, ccnet]:
        fp = obj.filepath

        if fp is None:
            continue

        hash_dist.run(lang=lang, datafile=fp, dataset=obj.dataset_name)

    if lang == "en":
        # compute fasttext palm classifier
        target_name = "palm"
        target_data = [
            wikipedia.filepath, books.filepath, openwebtext.filepath
        ]
    else:
        # for non english languages, we use wikipedia as target
        target_name = f"wikipedia"
        target_data = [wikipedia.filepath]

    trainer = FastTextTrainer(
        artifacts_dir=artifacts_dir,
        ccnet_data=ccnet.filepath,
        target_data=target_data,
        target_name=target_name,
        samples_per_class=classifiers_num_samples,
        lang=lang
    )
    trainer.run(logger=logger)

    logger.info(f"Finished preparing artifacts for {lang}")


if __name__ == '__main__':
    args = parse_arguments()
    main(artifacts_dir=args.artifacts_dir,
         cc_input=args.cc_input,
         cc_input_base_uri=args.cc_input_base_uri,
         cache_dir=args.cache_dir,
         overwrite=args.overwrite,
         lang=args.lang,
         max_workers=args.max_workers,
         endpoint_url=args.endpoint_url,
         dsir_num_samples=args.dsir_num_samples,
         dsir_feature_dim=args.dsir_feature_dim,
         classifiers_num_samples=args.classifiers_num_samples,
         max_samples_per_book=args.max_samples_per_book,
         max_paragraphs_per_book_sample=args.max_paragraphs_per_book_sample
         )
