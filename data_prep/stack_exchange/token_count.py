import os
import json
import tiktoken
from multiprocessing import Pool
from transformers import AutoTokenizer

# enc = tiktoken.get_encoding("r50k_base")
enc = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-6.9b-deduped",
  # "gpt2"
)

def get_token_count(qa_pair):
    # return len(enc.encode(qa_pair['text']))
    return len(enc.tokenize(qa_pair['text']))

LEMMA_DATA_DIR_SE_OUT = os.environ.get("LEMMA_DATA_DIR_SE_OUT", "./stackexchange/")

# if x is a file, not a dir
sites = [x for x in os.listdir(os.path.join(LEMMA_DATA_DIR_SE_OUT)) if os.path.isfile(os.path.join(LEMMA_DATA_DIR_SE_OUT, x))]

os.makedirs(os.path.join(LEMMA_DATA_DIR_SE_OUT, "token_counts"), exist_ok=True)

token_counts = {}
for site in sites:
    print(f"[INFO] Processing {site}...")
    with open(os.path.join(LEMMA_DATA_DIR_SE_OUT, site), "r") as f:
        qa_pairs = [json.loads(x) for x in f.readlines()]
    print(f"[INFO] Got {len(qa_pairs)} QA pairs for {site}.")
    # token count
    token_count = 0
    with Pool(24) as p:
        token_count = sum(p.map(get_token_count, qa_pairs))
    token_counts[site] = token_count
    print(f"[INFO] Got {token_count} tokens for {site}.")
    # write to file

with open(os.path.join(LEMMA_DATA_DIR_SE_OUT, "token_counts", "tokens.json"), "w") as f:
    json.dump(token_counts, f)