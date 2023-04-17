### StackExchange

Follow these instructions to create the Stack Exchange dataset.

#### Data Download

We download the StackExchange data dump from [here](https://archive.org/download/stackexchange). To start downloading,
run the following command in the `data_prep` folder:

```bash
mkdir -p data
python ./stack_exchange/download.py
```

Make sure you have `pandas`, `p7zip`, `lxml`, and `tqdm` installed.

You also need to download a fasttext model for language identification and store it in the models directory:

```bash
mkdir -p data
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data
```

#### Processing

> (From Llama paper): We include a dump of Stack Exchange, a website of high quality questions and answers that covers a
> diverse set of domains, ranging from computer science to chemistry. We kept the data from the 28 largest websites,
> removed the HTML tags from text and sorted the answers by score (from highest to lowest).

1) Filter largest 28 StackExchange sites.
    - The size of a site is defined as `len(posts)=len(questions)+len(answers)`, in other words, we don't count comments/users.
    - See the stats in [stats](data_stats/stackexchange.md).
    In order to get the count of each site, run the following command:

    ```bash
    # assume you are in the data_prep folder
    python ./stack_exchange/count.py
    ```
    It will write the count to `$LEMMA_DATA_DIR_SE/counts.json` (by default `$LEMMA_DATA_DIR_SE=data/stackexchange`).

2) Convert List of Posts (original dump format) to Q-As pairs.

    ```bash
    # assume you are in the data_prep folder
    python ./stack_exchange/filter.py
    ```
    It will write the Q-As pairs to `$LEMMA_DATA_DIR_SE/qa_pairs/` (by default `$LEMMA_DATA_DIR_SE=data/stackexchange`).

3) Post-processing: For each question, order answers by their score and remove HTML tags. We use [Beautifulsoup] for removing HTML tags. Furthermore, we replace all lists into "\n*".

      ```bash
      # assume you are in the data_prep folder
      python ./stack_exchange/postprocessing.py
      ```
      It will write the Q-As pairs to `$LEMMA_DATA_DIR_SE_OUT/` (by default `$LEMMA_DATA_DIR_SE_OUT=data`).
  
  This step may take a large amount of memory. If you run into OOM, please try split the data into multiple files and process them separately. For example, if you need to split stackoverflow into several chunks, you can run the following command:
  
  ```bash
  # assume you are in the $LEMMA_DATA_DIR_SE/qa_pairs/ folder
  split -l 1000000 stackoverflow.com-Posts.jsonl stackoverflow_
  ```

  Don't forget to move the original `stackoverflow.com-Posts.jsonl` file to another folder.

4) Token Counting

Finally, we count the number of tokens in the dataset.

```bash
# assume you are in the data_prep folder
python ./stack_exchange/token_count.py
```

It will write the token count to `$LEMMA_DATA_DIR_SE_OUT/token_counts/tokens.json` (by default `$LEMMA_DATA_DIR_SE_OUT=data`).
