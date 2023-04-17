import os
import json

LEMMA_DATA_DIR_SE_OUT = os.environ.get("LEMMA_DATA_DIR_SE_OUT", "./data/")

if __name__ == "__main__":
    with open(os.path.join(LEMMA_DATA_DIR_SE_OUT,"token_counts", "tokens.json"), "r") as f:
        counts = json.load(f)
    '''
    print a table of the counts
    '''
    print("|Idx|Site|Token Count|")
    print("|---|---|---|")
    for idx, (site, count) in enumerate(counts.items()):
        print(f"|{idx}|{site}|{count}|")
    print(f"|{len(counts.values())}|Total|{sum(counts.values())}|")