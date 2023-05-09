import glob, os
import json
import sys
import re
import hashlib
import gzip
import os

from multiprocessing import Pool

# Get all jobs.
# Each job corresponds to a file ends with .gz, with middle or head in it
#
jobs = []
os.chdir(sys.argv[1])
for file in glob.glob("*/*.gz"):
    if ("middle" in file or "head" in file) and "dedup" not in file:
        jobs.append(file)

print("TOTAL # JOBS:", len(jobs))


# For each row, run classifier and output
#    (text: [...], source, pred_label, pred_label_prob, wiki_prob)
#
def run(job):
    meta_fields = [
        'url', 'date_download', 'digest', 'length', 'nlines',
        'source_domain', 'title', 'original_nlines',
        'original_length', 'language', 'language_score',
        'perplexity'
    ]

    import fasttext
    model = fasttext.load_model("../fastText/model.bin")

    print(job)
    ofile = gzip.open(job + ".dedup.classifier.gz", "wt")
    ostat = open(job + ".dedup.classifier.gz.stat", "wt")
    line = 0
    for jstr in gzip.open(job + ".result", "rt"):
        result = json.loads(jstr)
        content = result["raw_content"]
        output = {}

        # run classifier
        text = " ".join(content.strip().splitlines())
        pred = model.predict(text)
        (pred_label, pred_prob) = pred
        pred_label = pred_label[0]

        wiki_prob = pred_prob[0]
        if pred_label == "__label__cc":
            wiki_prob = 1 - wiki_prob

        # collect metadata
        output["meta"] = {k: result[k] for k in meta_fields}  # ccnet metadata
        output["meta"]["pred_label"] = pred_label
        output["meta"]["pred_label_prob"] = pred_prob[0]
        output["meta"]["wiki_prob"] = wiki_prob
        output["meta"]["source"] = "cc/" + job + f"/line{line}"

        # text
        output["text"] = content
        line = line + 1

        nchars = len(content)
        ostat.write(f"{nchars}\t{wiki_prob}\n")
        ofile.write(json.dumps(output) + "\n")

    ofile.close()
    ostat.close()


with Pool(224) as p:
    p.map(run, jobs)
