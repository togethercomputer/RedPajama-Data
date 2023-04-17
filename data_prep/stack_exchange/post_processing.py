import re
import os
import sys
import json
import fasttext
from bs4 import BeautifulSoup
from multiprocessing import Pool

sys.path.append("./")

site_name = ""
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

def cleanhtml(raw_html):
    raw_html = raw_html.replace("<li>", "\n*")
    raw_html = raw_html.replace("</li>", "")
    raw_html = raw_html.replace("<ol>", "\n*")
    raw_html = raw_html.replace("</ol>", "")
    soup = BeautifulSoup(raw_html, "lxml")
    return soup.get_text()
    
class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "data/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        text = text.replace("\n", " ")
        predictions = self.model.predict(text, k=1) # returns top 2 matching languages
        return predictions[0][0].replace("__label__", "")

lang_id = LanguageIdentification()
LEMMA_DATA_DIR_SE = os.environ.get("LEMMA_DATA_DIR_SE", "./data/")
LEMMA_DATA_DIR_SE_OUT = os.environ.get("LEMMA_DATA_DIR_SE_OUT", "./data/")

os.makedirs(os.path.join(LEMMA_DATA_DIR_SE_OUT), exist_ok=True)

def process_qa_pair(pair):
    # sort answers by score
    if "answers" in pair:
        pair["answers"] = sorted(pair["answers"], key=lambda x: x["score"], reverse=True)
        answers = "\nA: ".join([ cleanhtml(x["text"]) for x in pair["answers"]])
        text = f"Q: { cleanhtml(pair['question']['text'])}\nA: {answers}"
    else:
        text = f"Q: { cleanhtml(pair['question']['text'])}"
    return {
        "text": text,
        "meta": {
            "language": lang_id.predict_lang(text),
            "url": f"https://{site_name}/questions/{pair['question']['id']}",
            "timestamp": "2023-03-29",
            "source": "stackexchange",
            "question_score": pair["question"]["score"],
        }
    }

# load qa_pairs
sites = [x for x in os.listdir(os.path.join(LEMMA_DATA_DIR_SE, "qa_pairs"))]

# if needed:
# sort sites such that stackoverflow is processed first - to understand the memory pressure
# if OOM -> split stackoverflow into multiple files
# this won't hurt the completeness of the data, as each line is self-contained
for site in sites:
    print(f"Processing {site}")
    results = []
    site_name = site.removesuffix(".jsonl")
    if "stackoverflow_part" in site_name:
        site_name = "stackoverflow.com"
    # load qa_pairs
    with open(os.path.join(LEMMA_DATA_DIR_SE, "qa_pairs", site), "r") as f:
        qa_pairs = [json.loads(x) for x in f.readlines()]
        # process html to text
        with Pool(24) as p:
            results = p.map(process_qa_pair, qa_pairs)

    print(f"Writing {len(results)} results to {os.path.join(LEMMA_DATA_DIR_SE_OUT, site)}")
    
    with open(os.path.join(LEMMA_DATA_DIR_SE_OUT, site), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")