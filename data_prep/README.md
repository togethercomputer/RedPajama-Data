## Data Preprocessing

We provide scripts and instructions to create the different slices of the dataset. Everything dataset will saved as jsonl files, following the format:

```
{"text": ..., "meta": {"url": "...", "timestamp": "...", "source": "...", "language": "...", ...}}
```

We follow the [Llama paper](https://arxiv.org/abs/2302.13971) and tried our best to reproduce its recipe. 

### Commoncrawl

We downlaod five dumps from Commoncrawl, and run the dumps through the official [`cc_net` pipeline](https://github.com/facebookresearch/cc_net).
We then deduplicate on the paragraph level, and filter out low quality text using a linear classifier trained to 
classify paragraphs as Wikipedia references or random Commoncrawl samples.

### C4

C4 is downloaded from Huggingface. The only preprocessing step is to bring the data into our own format.

### GitHub

The raw GitHub data is downloaded from Google BigQuery. We deduplicate on the file level and filter out low quality 
files and only keep projects that are distributed under the MIT, BSD, or Apache license.

### Wikipedia
We use the Wikipedia dataset available on Huggingface, which is based on the Wikipedia dump from 2023-03-20 and contains
text in 20 different languages. The dataset comes in preprocessed format, so that hyperlinks, comments and other 
formatting boilerplate has been removed.

### Gutenberg and Books3
The PG19 subset of the Gutenberg Project and Books3 datasets are downloaded from Huggingface. After downloading, we use 
simhash to remove near duplicates.

### ArXiv
ArXiv data is downloaded from Amazon S3 in the `arxiv` requester pays bucket. We only keep latex source files and 
remove preambles, comments, macros and bibliographies.

### Stackexchange
The Stack Exchange split of the dataset is download from the 
[Internet Archive](https://archive.org/download/stackexchange). Here we only keep the posts from the 28 largest sites,
remove html tags, group the posts into question-answer pairs, and order answers by their score.

## Token Counts

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
