# RedPajama-Data-v2: an Open Dataset with 30 Trillion Tokens for Training Large Language Models

<img width="500" src="docs/rpv2.png" />

This repository contains the code for the RedPajama-V2 dataset. For more information on the dataset, check out our
[blog post](https://together.ai/blog/redpajama-data-v2). The dataset is also available on
[HuggingFace](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2). For the code used for the
RedPajama-1T dataset, please refer to the `rp_v1` branch in this repo.

## Dataset

RedPajama-V2 is an open dataset for training large language models. The dataset includes over 100B text
documents coming from 84 CommonCrawl snapshots and processed using
the [CCNet](https://github.com/facebookresearch/cc_net) pipeline. Out of these, there are 30B documents in the corpus
that additionally come with quality signals, and 20B documents that are deduplicated.

### Document and Token Counts for the Annotated and deduplicated `head_middle` part of the dataset

The number of documents and tokens for the annotated and deduplicated `head_middle` part of the dataset is shown in the
table below.

|       | # Documents | Estimated Token count (deduped) |
|-------|-------------|---------------------------------|
| en    | 14.5B       | 20.5T                           |
| de    | 1.9B        | 3.0T                            |
| fr    | 1.6B        | 2.7T                            |  
| es    | 1.8B        | 2.8T                            |
| it    | 0.9B        | 1.5T                            |
| Total | 20.8B       | 30.4T                           |

### Languages

English, German, French, Italian, Spanish

## Setup

### Configuration

Copy the file `configs/rp_v2.0.conf` to e.g. `configs/default.conf` and configure the environment variables.
These will be used throughout the pipeline.

### Build Docker image

To run with docker, build the docker image using

```bash
. configs/default.conf
cd app
docker build -t "${DOCKER_REPO}:" .

```

Also, make sure you have `s5cmd` installed and your S3 profile configured so that you can pull data from an S3 bucket.

You can run the steps of the pipeline without any containerized environment. However, the running scripts assume you
have a docker and apptainer installation.

## Running the Pipeline

The pipeline is composed of three steps, namely 1) preparing artifacts, 2) computing quality signals, and 3)
deduplication.

**Important:** In case you are not running steps (1) and (2) with the provided scripts (i.e., docker containers built with the provided Dockerfile), make sure to set the `PYTHONHASHSEED` environment variable to a consistent value (e.g., 42) using
```bash
export PYTHONHASHSEED=42
```
This is to ensure consistency of hash functions used in the computation of DSIR weights.

### 1. Create Artifacts

This part of the pipeline creates the artifacts that are used in the subsequent steps. This includes building quality
classifiers, training bag-of-ngram generative models for importance weight computation, fetching the list of bad words
from the [LDNOOBW repo](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), and fetching
the most recent list of blacklisted urls from the [UT1 blacklist](https://dsi.ut-capitole.fr/blacklists/).

As a first step, download the english wikipedia reference classifier
from [here](https://data.together.xyz/redpajama-data-v2/v1.0.0/artifacts/wikiref.model.bin) and place it
in `${DATA_ROOT}/wikiref-model/en/en-model.bin`. This is the same fasttext classifier that was used in RedPajama-V1.

To create the remaining artifacts, make sure that the environment variables are set in the config file. Then, from
the root directory of the repository, run

```bash
bash scripts/run_prep_artifacts.sh \
  --config configs/rp_v2.0.conf \
  --listings /path/to/listings/file.txt\
  --max_workers 32
```

where `/path/to/listings/file.txt` is a file that contains the keys to the ccnet data that you want to process
(e.g., `2023-06/0000/en_head.json.gz`).

You can set the `max_workers` flag to the number of parallel processes you want to use.

This step will generate an id which you can store in the environment variable `ARTIFACTS_ID` for the next step.

### 2. Compute Quality Signals

The second step of the pipeline compute the quality signals, including the minhash signatures to run fuzzy deduplication
in the subsequent step. To run this step, make sure the environment variables are set in the config file. Then, from
the root directory of the repository, run

```bash
bash scripts/apptainer_run_quality_signals.sh \
  --config configs/rp_v2.0.conf \
  --dump_id "2022-49" \
  --input_base_uri "file:///path/to/data/root" \
  --output_base_uri "file:///path/to/outout/data/root" \
  --max_docs -1
```

### 3. Deduplication

The third component of the pipeline consists of deduplication steps. Here we provide code to run exact and fuzzy
deduplication.

#### Exact Deduplication using a Bloomfilter

Content based deduplication is implemented in `app/src/bloomfilter.py`. It can be run independently of the
previous step, but the data needs to stored in an S3 bucket. For this step, from the `app` directory, run:

```bash
python3 app/src/bloomfilter.py \
  --listings /path/to/listings/file.txt \
  --input_base_uri "s3://path/to/ccnet/data" \
  --output_dir "/path/to/output" \
  --s3_profile "..." \
  --endpoint_url "..." \
  --parallel_readers 32 \
  --batch_size 10 \
  --capacity "..." \
  --error_rate "..."
```

It is important to choose the correct capacity (i.e., > #documents), since otherwise the `error_rate` will not be
guaranteed and more false positives will appear. The implementation is based on the
[pybloomfiltermmap3](https://github.com/prashnts/pybloomfiltermmap3) library.

#### Fuzzy Deduplication with Locality Sensitive Hashing

In the third step of the pipeline, we run locality sensitive hashing on the minhash signatures generated in the first
step. To run this step, make sure that you use the same configuration as in the quality signals step. Then, from
the root directory of the repository, run

```bash
bash scripts/apptainer_run_lsh.sh \
  --config configs/rp_v2.0.conf \
  --dump_id "2022-49" \
  --input_base_uri "file:///path/to/data/root" \
  --output_dir "/path/to/output" \
  --similarity "<similarity_threshold>" \
  --listings "/minhash/listings/file.txt" \
  --max_docs -1
```

The implementation is based on polars and was tested with 200M documents on a 64 core machine with 500G of RAM.

## Summary of Quality Signals

The second step of this pipeline computes the following set of quality signals. We hope to grow this list further over
time as more signals are developed.

#### Quality Annotations

| Annotation Tag                                 | Description                                                                                                                                                                                                                                                                                                                                                                                                          | Category         | Reference                                                                                                                     |
|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------------------|
| ccnet_bucket                                   | head, middle or tail bucket of the perplexity score                                                                                                                                                                                                                                                                                                                                                                  | CCNet            | [CCNet](https://github.com/facebookresearch/cc_net)                                                                           |
| ccnet_language_score                           | score of the language identification model                                                                                                                                                                                                                                                                                                                                                                           | CCNet            | [CCNet](https://github.com/facebookresearch/cc_net)                                                                           |
| ccnet_length                                   | number of characters                                                                                                                                                                                                                                                                                                                                                                                                 | CCNet            | [CCNet](https://github.com/facebookresearch/cc_net)                                                                           |
| ccnet_nlines                                   | number of lines                                                                                                                                                                                                                                                                                                                                                                                                      | CCNet            | [CCNet](https://github.com/facebookresearch/cc_net)                                                                           |
| ccnet_original_length                          | number of characters before in-document line deduplication                                                                                                                                                                                                                                                                                                                                                           | CCNet            | [CCNet](https://github.com/facebookresearch/cc_net)                                                                           |
| ccnet_original_nlines                          | number of lines before in-document line deduplication                                                                                                                                                                                                                                                                                                                                                                | CCNet            | [CCNet](https://github.com/facebookresearch/cc_net)                                                                           |
| ccnet_perplexity                               | perplexity of an LM trained on Wikipedia                                                                                                                                                                                                                                                                                                                                                                             | CCNet            | [CCNet](https://github.com/facebookresearch/cc_net)                                                                           |
| rps_doc_books_importance                       | Given a bag of {1,2}-wordgram model trained on Books p, and a model trained on the source domain q, This is the logarithm of the ratio p(doc)/q(doc).                                                                                                                                                                                                                                                                | ML Heuristics    | [Importance Resampling (Xie et al.)](https://arxiv.org/abs/2302.03169)                                                        |
| rps_doc_openwebtext_importance                 | Given a bag of {1,2}-wordgram model trained on OpenWebText p, and a model trained on the source domain q, this is the logarithm of the ratio p(doc)/q(doc).                                                                                                                                                                                                                                                          | ML Heuristics    | [Importance Resampling (Xie et al.)](https://arxiv.org/abs/2302.03169)                                                        |
| rps_doc_wikipedia_importance                   | Given a bag of {1,2}-wordgram model trained on Wikipedia articles p, and a model trained on the source domain q, this is the logarithm of the ratio p(doc)/q(doc).                                                                                                                                                                                                                                                   | ML Heuristics    | [Importance Resampling (Xie et al.)](https://arxiv.org/abs/2302.03169)                                                        |
| rps_doc_ml_wikiref_score                       | Fasttext classifier prediction for the document being a Wikipedia reference. This is the same fasttext model used in the RedPajama-1T dataset. Only applies to English data..                                                                                                                                                                                                                                        | ML Heuristics    | [LLaMA](https://arxiv.org/abs/2302.13971), [RedPajama-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) |
| rps_doc_ml_palm_score                          | Fasttext classifier prediction for the document being a Wikipedia article, OpenWebText sample or a RedPajama-V1 book. Only for English data.                                                                                                                                                                                                                                                                         | ML Heuristics    | [PALM](https://arxiv.org/abs/2204.02311), [GLaM](https://arxiv.org/abs/2112.06905)                                            |
| rps_doc_ml_wikipedia_score                     | Fasttext classifier prediction for the document being a Wikipedia article. This is used for non-English data                                                                                                                                                                                                                                                                                                         | ML Heuristics    | -                                                                                                                             |
| rps_doc_curly_bracket                          | The ratio between the number of occurrences of '{' or '}' and the number of characters in the raw text.                                                                                                                                                                                                                                                                                                              | Natural Language | [C4](https://arxiv.org/abs/1910.10683)                                                                                        |
| rps_doc_frac_all_caps_words                    | The fraction of words in the content that only consist of uppercase letters. This is based on the raw content.                                                                                                                                                                                                                                                                                                       | Natural Language | [Pretrainer’s Guide](https://arxiv.org/abs/2305.13169)                                                                        |
| rps_doc_frac_lines_end_with_ellipsis           | The fraction of lines that end with an ellipsis, where an ellipsis is defined as either "..." or "…".                                                                                                                                                                                                                                                                                                                | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_no_alph_words                     | The fraction of words that contain no alphabetical character.                                                                                                                                                                                                                                                                                                                                                        | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_lorem_ipsum                            | The ratio between the number of occurrences of 'lorem ipsum' and the number of characters in the content after normalisation.                                                                                                                                                                                                                                                                                        | Natural Language | [C4](https://arxiv.org/abs/1910.10683)                                                                                        |
| rps_doc_mean_word_length                       | The mean length of words in the content after normalisation.                                                                                                                                                                                                                                                                                                                                                         | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_stop_word_fraction                     | The ratio between the number of stop words and the number of words in the document. Stop words are obtained from the [stopwords-json](https://github.com/6/stopwords-json) repo.                                                                                                                                                                                                                                     | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_symbol_to_word_ratio                   | The ratio of symbols to words in the content.. Symbols are defined "#", "...", and "…".                                                                                                                                                                                                                                                                                                                              | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_unique_words                      | The fraction of unique words in the content. This is also known as the degeneracy of a text sample. Calculated based on the normalised content.                                                                                                                                                                                                                                                                      | Natural Language | [Pretrainer’s Guide](https://arxiv.org/abs/2305.13169)                                                                        |
| rps_doc_unigram_entropy                        | The entropy of the unigram distribution of the content. This measures the diversity of the content and is computed using sum(-x / total * log(x / total)) where the sum is taken over counts of unique words in the normalised content.                                                                                                                                                                              | Natural Language | -                                                                                                                             |
| rps_doc_word_count                             | The number of words in the content after normalisation.                                                                                                                                                                                                                                                                                                                                                              | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_lines_ending_with_terminal_punctution_mark | Indicates whether a line ends with a terminal punctuation mark. A terminal punctation mark is defined as one of: ".", "!", "?", "”".                                                                                                                                                                                                                                                                                 | Natural Language | [C4](https://arxiv.org/abs/1910.10683)                                                                                        |
| rps_lines_javascript_counts                    | The number of occurrences of the word "javascript" in each line.                                                                                                                                                                                                                                                                                                                                                     | Natural Language | [C4](https://arxiv.org/abs/1910.10683)                                                                                        |
| rps_lines_num_words                            | The number of words in each line. This is computed based on the normalised text.                                                                                                                                                                                                                                                                                                                                     | Natural Language | [C4](https://arxiv.org/abs/1910.10683) , [RefinedWeb](https://arxiv.org/abs/2306.01116)                                       |
| rps_lines_numerical_chars_fraction             | The ratio between the number of numerical characters and total number of characters in each line. This is based on the normalised content.                                                                                                                                                                                                                                                                           | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116)                                                                                |
| rps_lines_start_with_bulletpoint               | Whether the lines that start with a bullet point symbol. The following set of unicodes are considered a bullet point: \u2022 (bullet point), \u2023 (triangular bullet point), \u25B6 (black right pointing triangle), \u25C0 (black left pointing triangle), \u25E6 (white bullet point), \u25A0 (black square), \u25A1 (white square), \u25AA (black small square), \u25AB (white small square), \u2013 (en dash). | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_lines_uppercase_letter_fraction            | The ratio between the number of uppercase letters and total number of characters in each line. This is based on the raw text.                                                                                                                                                                                                                                                                                        | Natural Language | [RefinedWeb](https://arxiv.org/abs/2306.01116)                                                                                |
| rps_doc_num_sentences                          | The number of sentences in the content. This is calculated using the regular expression `r'\b[^.!?]+[.!?]*'`.                                                                                                                                                                                                                                                                                                        | Natural Language | [C4](https://arxiv.org/abs/1910.10683)                                                                                        |
| rps_doc_frac_chars_dupe_10grams                | The fraction of characters in duplicate word 10grams. This operates on the lower-cased, punctuation removed content. It is also ensured that characters in overlapping ngrams are only counted once.                                                                                                                                                                                                                 | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_chars_dupe_5grams                 | The fraction of characters in duplicate word 5grams.                                                                                                                                                                                                                                                                                                                                                                 | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_chars_dupe_6grams                 | The fraction of characters in duplicate word 6grams.                                                                                                                                                                                                                                                                                                                                                                 | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_chars_dupe_7grams                 | The fraction of characters in duplicate word 7grams.                                                                                                                                                                                                                                                                                                                                                                 | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_chars_dupe_8grams                 | The fraction of characters in duplicate word 8grams.                                                                                                                                                                                                                                                                                                                                                                 | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_chars_dupe_9grams                 | The fraction of characters in duplicate word 9grams.                                                                                                                                                                                                                                                                                                                                                                 | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_chars_top_2gram                   | The fraction of characters in the top word 2gram.                                                                                                                                                                                                                                                                                                                                                                    | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_chars_top_3gram                   | The fraction of characters in the top word 3gram.                                                                                                                                                                                                                                                                                                                                                                    | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_frac_chars_top_4gram                   | The fraction of characters in the top word 4gram.                                                                                                                                                                                                                                                                                                                                                                    | Repetitiveness   | [RefinedWeb](https://arxiv.org/abs/2306.01116), [Gopher](https://arxiv.org/abs/2112.11446)                                    |
| rps_doc_ldnoobw_words                          | The number of sequences of words that are contained in the List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words blocklist. The blocklist is obtained from the [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) repo.                                                                                                                                                     | toxicity         | [C4](https://arxiv.org/abs/1910.10683)                                                                                        |
| rps_doc_ut1_blacklist                          | A categorical id corresponding to the list of categories of the domain of the document. Categories are obtained from the UT1 blacklist. The list is obtained from [UT-Capitole](https://dsi.ut-capitole.fr/blacklists/).                                                                                                                                                                                             | toxicictiy       | [RefinedWeb](https://arxiv.org/abs/2306.01116)                                                                                |
| minhash_signature_0.7                          | Banded minhash signature of the document, for fuzzy deduplication at Jaccard similarity 0.7. The signature is based on 128 hash functions and grouped into 14 bands and 9 rows for LSH.                                                                                                                                                                                                                              | Deduplication    |
| minhash_signature_0.8                          | Banded minhash signature of the document, for fuzzy deduplication at Jaccard similarity 0.8. The signature is based on 128 hash functions and grouped into 9 bands and 13 rows for LSH.                                                                                                                                                                                                                              | Deduplication    |
| minhash_signature_0.9                          | Banded minhash signature of the document, for fuzzy deduplication at Jaccard similarity 0.9. The signature is based on 128 hash functions and grouped into 5 bands and 25 rows for LSH..                                                                                                                                                                                                                             | Deduplication    |
| minhash_signature_1.0                          | Banded minhash signature of the document, for fuzzy deduplication at Jaccard similarity 1.0. The signature is based on 128 hash functions and grouped into 1 band and 128 rows for LSH.                                                                                                                                                                                                                              | Deduplication    |

## Acknowledgements

We are appreciative to so many partners and collaborators that together are pushing forward the frontier of open LLM
models.

- Thank you to the OLMo team at AI2 and friends at OpenGPT-X for the insightful discussions about datasets and data
  quality! Also for everyone who builds on the RedPajama dataset, including Cerebras for their SlimPajama efforts, and
  the over 500 models built on RedPajam to date by the open-source AI community.
- We are grateful to the great team at EleutherAI for paving the path on open training datasets with The Pile and for
  open-sourcing code we use in training some of the RedPajama models.
- Thank you to our partners of RedPajama-v1, including Ontocord.ai, MILA Québec AI Institute, ETH DS3Lab, Université de
  Montréal, Stanford Center for Research on Foundation Models (CRFM), Stanford Hazy Research research group and LAION.

## License

```
Copyright 2023 Together Computer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

For full terms, see the LICENSE file. If you have any questions, comments, or concerns about licensing
please [contact us](https://www.together.ai/contact).

For the dataset itself, please refer to
the [Common Crawl Foundation Terms of Use](https://commoncrawl.org/terms-of-use).

To cite RedPajama, please use:

```
@article{weber2024redpajama,
	title   = {RedPajama: an Open Dataset for Training Large Language Models},
	author  = {Maurice Weber and Daniel Y. Fu and Quentin Anthony and Yonatan Oren and Shane Adams and Anton Alexandrov and Xiaozhong Lyu and Huu Nguyen and Xiaozhe Yao and Virginia Adams and Ben Athiwaratkun and Rahul Chalamala and Kezhen Chen and Max Ryabinin and Tri Dao and Percy Liang and Christopher Ré and Irina Rish and Ce Zhang},
	journal = {NeurIPS Datasets and Benchmarks Track},
	year    = 2024,
}
```

