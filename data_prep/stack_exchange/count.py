import os
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

LEMMA_DATA_DIR_SE = os.environ.get("LEMMA_DATA_DIR_SE", "./data/stack_exchange/")

def get_sites_count(path=LEMMA_DATA_DIR_SE):
    sites = os.listdir(path)
    sites = [x for x in sites if x.endswith(".xml")]
    counts = {}
    for site in tqdm(sites):
        if site == ".DS_Store":
            continue
        # read the file
        with open(os.path.join(path, site), "r") as f:
            # read # lines
            count = sum(1 for line in f)
            counts[site] = count-3 # subtract the header
    # sort the counts
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    return counts


if __name__ == "__main__":
    counts = get_sites_count()
    '''
    print a table of the counts
    '''
    print("|Idx|Site|Count|")
    print("|---|---|---|")
    # take the first 28 sites
    for idx, (site, count) in enumerate(counts.items()):
        if idx < 28:
            print(f"|{idx}|{site}|{count}|")
    # write to file
    with open(os.path.join(LEMMA_DATA_DIR_SE, "counts.json"), "w") as f:
        json.dump(counts, f)