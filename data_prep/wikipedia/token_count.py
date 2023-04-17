import os
import json
from multiprocessing import Pool
from transformers import AutoTokenizer

print("start loading!")
enc = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-6.9b-deduped",
)
print("end loading!")

def get_token_count(qa_pair):
    return len(enc.tokenize(qa_pair['text']))

LEMMA_DATA_DIR_SE_OUT = "./data/wikipedia/"
sites = [x for x in os.listdir(os.path.join(LEMMA_DATA_DIR_SE_OUT)) if os.path.isfile(os.path.join(LEMMA_DATA_DIR_SE_OUT, x))]

os.makedirs(os.path.join(LEMMA_DATA_DIR_SE_OUT, "token_counts"), exist_ok=True)

token_counts = {}
for site in sites:
    print(f"[INFO] Processing {site}...")
    with open(os.path.join(LEMMA_DATA_DIR_SE_OUT, site), "r") as f:
        qa_pairs = [json.loads(x) for x in f.readlines()]
    print(f"[INFO] Got {len(qa_pairs)} wikipedia pages for {site}.")

    token_count = 0
    with Pool(100) as p:
        token_count = sum(p.map(get_token_count, qa_pairs))
    token_counts[site] = token_count
    print(f"[INFO] Got {token_count} tokens for {site}.")

with open(os.path.join(LEMMA_DATA_DIR_SE_OUT, "token_counts", site), "w") as f:
    json.dump(token_counts, f)