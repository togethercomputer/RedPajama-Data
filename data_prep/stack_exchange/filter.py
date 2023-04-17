import os
import json
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm
sys.path.append("./")
from src.stack_exchange.count import get_sites_count

LEMMA_DATA_DIR_SE = os.environ.get("LEMMA_DATA_DIR_SE", "./data/")

if os.path.exists(os.path.join(LEMMA_DATA_DIR_SE, "counts.json")):
    with open(os.path.join(LEMMA_DATA_DIR_SE, "counts.json"), "r") as fp:
        counts = json.load(fp)
else:
    print("[INFO] Getting counts for sites...")
    counts = get_sites_count(LEMMA_DATA_DIR_SE)
    # write this to a file
    with open(os.path.join(LEMMA_DATA_DIR_SE, "counts.json"), "w") as f:
        json.dump(counts, f)

# take first 28
sites = list(counts.keys())[:28]
os.makedirs(os.path.join(LEMMA_DATA_DIR_SE, "parents"), exist_ok=True)

def process_site(site):
    parents = {}
    qa_pairs = []
    print(f"[INFO] Processing {site}...")
    # first get the parents dump
    if os.path.exists(os.path.join(LEMMA_DATA_DIR_SE, "parents", site)):
        with open(os.path.join(LEMMA_DATA_DIR_SE, "parents", site), "r") as f:
            parents = json.load(f)
    else:        
        with open(os.path.join(LEMMA_DATA_DIR_SE, site), "r") as f:
            for i, line in enumerate(tqdm(f, total=counts[site])):
                # first 2 lines are header
                # e.g., counts = 2: total=5 lines, 2,3 are data
                # last line is footer
                if i>1 and i<=counts[site]+1:
                    root = ET.fromstring(line)
                    if "ParentId" in root.attrib:
                        # this is an answer
                        if root.attrib["ParentId"] not in parents:
                            parents[root.attrib["ParentId"]] = []
                        parents[root.attrib["ParentId"]].append({
                            "id": root.attrib["Id"],
                            "text": root.attrib["Body"],
                            "score": root.attrib["Score"]
                        })
        # write parents to file
        with open(os.path.join(LEMMA_DATA_DIR_SE, "parents", site), "w") as f:
            json.dump(parents, f)

    print(f"[INFO] Got {len(parents)} questions for {site}.")
    # now we have the Q-A pairs
    # now we need to get the texts

    with open(os.path.join(LEMMA_DATA_DIR_SE, site), "r") as f:
        for i, line in enumerate(tqdm(f, total=counts[site])):
            if i>1 and i<=counts[site]+1:
                root = ET.fromstring(line)
                if "ParentId" not in root.attrib:
                    post_id = root.attrib["Id"]
                    if post_id in parents:
                        # this is a question
                        qa_pairs.append({
                            "question": {
                                "id": post_id,
                                "text": f"{root.attrib['Title']} {root.attrib['Body']}",
                                "score": root.attrib["Score"]
                            },
                            "answers": parents[post_id]
                        })
                    else:
                        if "Title" in root.attrib:
                            # if there's a title => then a valid question
                            body = root.attrib["Body"] if "Body" in root.attrib else ""
                            score = root.attrib["Score"] if "Score" in root.attrib else 0
                            qa_pairs.append({
                                "question": {
                                    "id": post_id,
                                    "text": f"{root.attrib['Title']} {body}",
                                    "score": score
                                },
                            })
    # write qa_pairs to file
    print(f"[INFO] Writing {site} to file...")
    os.makedirs(os.path.join(LEMMA_DATA_DIR_SE, "qa_pairs"), exist_ok=True)
    with open(os.path.join(LEMMA_DATA_DIR_SE, "qa_pairs", site.removesuffix(".xml")+".jsonl"), "w") as f:
        for qa_pair in qa_pairs:
            f.write(json.dumps(qa_pair)+"\n")

for each in sites:
    process_site(each)