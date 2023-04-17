from datasets import load_dataset
import pickle

def get_data(lan = "en", date = "20230320"):
    wiki_dataset = load_dataset("wikipedia", language=lan, date=date, beam_runner="DirectRunner")

    for split, dataset in wiki_dataset.items():
        dataset.to_json(f"./data/wikipedia/wiki_{lan}_{date}_{split}.jsonl")

    print("Finish Downloading %s %s. There are total %d pages."%(lan, date, len(dataset["id"])))

if __name__ == "__main__":
    language = ["bg", "ca", "cs", "da", "de", "en", "es", "fr", "hr", "hu", "it", "nl", "pl", "pt", "ro", "ru", "sl", "sr", "sv", "uk"]
    for lan in language:
        for date in ["20230320"]:
            get_data(lan, date)