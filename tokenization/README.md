# Tokenization

We an example of tokenizing the RedPajama dataset using the GPT-NeoX tokenizer.
These commands have been tested on this [commit](https://github.com/EleutherAI/gpt-neox/tree/831af97ecc071b345615f45982f15c0d32a887b8) of the `gpt-neox` repo.
This pipeline will produce slices with the following token counts:

| Dataset       | Token Count |
|---------------|-------------|
| Commoncrawl   | 878 Billion        |
| C4            | 175 Billion        |
| GitHub        | 59 Billion         |
| Books         | 26 Billion         |
| ArXiv         | 28 Billion         |
| Wikipedia     | 24 Billion         |
| StackExchange | 20 Billion         |
| Total         | 1.2 Trillion      |

## Installation

```
git clone https://github.com/EleutherAI/gpt-neox.git
pip install -r requirements/requirements.txt

wget https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/20B_tokenizer.json
```

## Tokenization

Assuming that the root of the data are in `DATA`:

```
cd gpt-neox

# Common Crawl Dumps, change for each dump (2019-30, 2020-05, 2021-04, 2022-05, 2023-06)
FILES=$(ls DATA/common_crawl/2023-06/*.jsonl.zst | tr '\n' ' ' | sed 's/ /,/g')
FILES=${FILES:0:-1}

python tools/preprocess_data.py \
    --input files \
    --output-prefix DATA/common_crawl/2023-06/tokenized \
    --vocab 20B_tokenizer.json \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --jsonl-keys text \
    --workers 64

# C4
FILES=$(ls DATA/c4/c4-train* | tr '\n' ' ' | sed 's/ /,/g')
FILES=${FILES:0:-1}

python tools/preprocess_data.py \
    --input files \
    --output-prefix DATA/c4/tokenized \
    --vocab 20B_tokenizer.json \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --jsonl-keys text \
    --workers 64

# Github
FILES=$(ls DATA/github/*.jsonl | tr '\n' ' ' | sed 's/ /,/g')
FILES=${FILES:0:-1}

python tools/preprocess_data.py \
    --input files \
    --output-prefix DATA/github/tokenized \
    --vocab 20B_tokenizer.json \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --jsonl-keys text \
    --workers 64

# Books
python tools/preprocess_data.py \
    --input DATA/book/book.jsonl \
    --output-prefix DATA/book/tokenized \
    --vocab 20B_tokenizer.json \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --jsonl-keys text \
    --workers 64

# arXiv
FILES=$(ls DATA/arxiv/*.jsonl | tr '\n' ' ' | sed 's/ /,/g')
FILES=${FILES:0:-1}

python tools/preprocess_data.py \
    --input files \
    --output-prefix DATA/arxiv/tokenized \
    --vocab 20B_tokenizer.json \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --jsonl-keys text \
    --workers 64

# Wikipedia
python tools/preprocess_data.py \
    --input DATA/wikipedia/wiki.jsonl \
    --output-prefix DATA/wikipedia/tokenized \
    --vocab 20B_tokenizer.json \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --jsonl-keys text \
    --workers 64

# StackExchange
python tools/preprocess_data.py \
    --input DATA/stackexchange/stackexchange.jsonl \
    --output-prefix DATA/stackexchange/tokenized \
    --vocab 20B_tokenizer.json \
    --tokenizer-type HFTokenizer \
    --append-eod \
    --jsonl-keys text \
    --workers 64

```

Each of these commands will produce two files: `DATA/<dataset>/tokenized_text_document.bin` and `DATA/<dataset>/tokenized_text_document.idx`, which can be used for training with Megatron.
If you'd like to use a different tokenizer, you can replace the `--vocab` and `--tokenizer-type` arguments with the appropriate values (see the GPT-NeoX [README](https://github.com/EleutherAI/gpt-neox/tree/831af97ecc071b345615f45982f15c0d32a887b8#using-custom-data) for more details).

## Counting Tokens

The `count_tokens.py` script in this repo can be used to sanity check the tokenization process:

```
PYTHONPATH=PATH_TO_GPT_NEOX python count_tokens.py \
    DATA/<dataset>/tokenized_text_document \
    DATA/<dataset>/tokenized_text_document.stat
```

This command generates a file `tokenized_text_document.stat` that contains the number of tokens in each document in the dataset, and prints out the total number of tokens in the dataset.

