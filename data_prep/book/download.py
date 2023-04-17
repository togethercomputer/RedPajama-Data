from datasets import load_dataset
import pickle

book_dataset = load_dataset("the_pile_books3")

for split, dataset in book_dataset.items():
    dataset.to_json(f"./data/book/books3-{split}.jsonl")

pg19_dataset = load_dataset("pg19")

for split, dataset in pg19_dataset.items():
    dataset.to_json(f"./data/book/pg19-{split}.jsonl")