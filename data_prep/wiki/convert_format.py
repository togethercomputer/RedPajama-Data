import os
import json

LEMMA_DATA_DIR_SE_OUT = "./data/wikipedia/"
LEMMA_DATA_SAVE_DIR = "./data/wikipedia/wiki-full.jsonl"

files = [x for x in os.listdir(os.path.join(LEMMA_DATA_DIR_SE_OUT)) if os.path.isfile(os.path.join(LEMMA_DATA_DIR_SE_OUT, x))]
files.sort()

with open(LEMMA_DATA_SAVE_DIR, "w") as fw:
    for file in files:
        lan = file.split("_")[1]
        date = file.split("_")[2]
        print("Now proceeding %s"%file, lan, date)

        with open(os.path.join(LEMMA_DATA_DIR_SE_OUT, file), "r") as f:
            lines = f.readlines()
            for line in lines:
                now = json.loads(line)
                new = {"text": now["text"], "meta": {"title": now["title"], "url": now["url"], "language": lan, "timestamp": date}}
                fw.write(json.dumps(new) + "\n")
        