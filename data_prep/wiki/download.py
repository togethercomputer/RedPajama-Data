import argparse
from datasets import load_dataset
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None,
                    help="Path to the wikipedia data directory.")
args = parser.parse_args()

LANGUAGES = [
    "bg", "ca", "cs", "da", "de", "en", "es", "fr", "hr", "hu",
    "it", "nl", "pl", "pt", "ro", "ru", "sl", "sr", "sv", "uk"
]

DUMP_DATE = "20230320"


def get_data(lan, date, data_dir: pathlib.Path):
    wiki_dataset = load_dataset(
        "wikipedia", language=lan, date=date, beam_runner="DirectRunner"
    )

    for split, dataset in wiki_dataset.items():
        tgt_fp = data_dir / f"wiki_{lan}_{date}_{split}.jsonl"
        dataset.to_json(tgt_fp)

    print("Finished Downloading %s %s. There are total %d pages." % (
        lan, date, len(dataset["id"])))


if __name__ == "__main__":
    if args.data_dir is None:
        raise ValueError("missing arg --data_dir.")

    for lang in LANGUAGES:
        get_data(lang, DUMP_DATE, data_dir=pathlib.Path(args.data_dir))
