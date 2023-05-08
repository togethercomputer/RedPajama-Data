## CommonCrawl

The processing of the CommonCrawl dataset contains the following stages:

  - Quality Filter using `cc-net`
  - Deduplication
  - Quality Classifier (Random CommonCrawl page vs. Wikipedia References)

### Quality Filter using `cc-net`

We use the [`cc-net`](https://github.com/facebookresearch/cc_net) pipeline out of the box to preprocess English-text web pages from five CommonCrawl dumps: `2019-30`, `2020-05`, `2021-04`, `2022-05`, and `2023-06`.
This pipeline downloads the data, removes duplicates within a dump, detects language, and applies a language model to compute perplexity and filter out low-quality pages.

Here are steps to reproduce the pipeline on a single machine with a large amount of RAM, using the `cc_net` clone in this folder:

```
# Installation
cd `cc_net`
mkdir data

sudo apt-get update
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
make install
make lang=en dl_lm

# Run CC Net on the 2023-06 dump
python -m cc_net --dump 2023-06 --task_parallelism 20 --num_shards 5000 -l en --mine_num_processes 20 --hash_in_mem 1
```

### Deduplication

Deduplication happens in two stages. For each `.gz` file output by `cc-net`, this stage will generate two files:

```
A.gz
  => A.gz.dedup    // intermediate result for deduplication
  => A.gz.result   // content after deduplication
```

In the first stage, we scan all ".gz" files (only the "middle" and "head" partition) and extract `(URL, digest)` pairs from each file:

```
python dedup/dedup_phase1.py FILE_DIR
```

where `FILE_DIR` is where the outputs of `cc-net` are.

In the second stage, we load all pairs of `(URL, digest)` to memory and conduct deduplication using `digest`, and then scan all ".gz" files and keep only those that are unique:

```
python dedup/dedup_phase2.py FILE_DIR
```

### Quality Classifier

Quote from the LLaMA paper:
> we trained a linear model to classify pages used as references in Wikipedia v.s. randomly sampled pages, and discarded pages not classified as references.

#### Crawling Wikipedia References
We downloaded the most recent English Wikipedia dump available by April 1, 2023 from https://dumps.wikimedia.org/enwiki/20230401. Unizpping the large bz2 folder with the following script should leave you with an XML file.

``` 
bzip2 -dk enwiki-20230401-pages-articles-multistream.xml.bz2 
``` 

Extract all the reference URLs from the XML and put them in a newline-delimited text file with ``` extract_urls.py```. You can specify input and output file. The default output file is `extracted_urls.txt`.

In `extracted_urls.txt`, we provide 38M URLs that are processed from the Wikipedia dump. We early stop this process to only keep 300K pages.

```
wget –-timeout=5 -i urls_random.txt --warc-file=warc_wikipedia.warc -O /dev/null
```

We then run the same `cc-net` pipeline on `warc_wikipedia.warc`, which produces `warc_wikipedia.warc.wet`.

The next step is to random sample the same number of CommonCrawl pages as Wikipedia References:

```
python classifier/create_corpus.py > data_train
```

#### Train a Classifier for Filtering
We then train a classifier using fastText:

```
fasttext supervised -input data_train -output model
```

We then classify each CommonCrawl webpage using the trained classifier. The model weights can be downloaded [here](https://drive.google.com/file/d/1DnsfpWWE0jFPCoYe6clwqb3Ub5Ac92s1/view?usp=share_link). 

```
python classifier/classify.py
```

This will create:

```
A.gz.result              // content after deduplication
A.gz.dedup.classifier.gz // result with classifier probability
```

Finally, we filter out all documents that have score less than `0.25`:
```
for file in $(ls $FILE_DIR/*.gz); do bash filter/cc_classifier.sh "$file" & done
```
