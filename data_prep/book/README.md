### Books

Follow these instructions to create the Books dataset.

We use the following two datasets: the pg19 subset of Gutenberg and the-pile-books3 for Books3, both downloaded from
Huggingface.

To download the dataset, run the following commands.

```bash
mkdir -p data/book/
python ./book/download.py
```

The data in JSON format should be around 119GB after downloading. Using the script `dedup.py`, we remove duplicates
when combining those two datasets. For deduplication, SIMHASH is used to map each book to a hash value and to remove
books within a certain distance. Run the following command to install the SIMHASH library.

```
pip install simhash
```

As a default setting, we compute `w = 6` grams of the entire book text with `l = 0` and find duplications with the
hyperparameter `k = 5`. We run the deduplication process on `n = 100` processors. To change the settings, you can
change the hyperparameters `-w`, `-k`, `-l`, and `-n`.

```bash
mkdir -p data/book/split/
split -a 3 -d -l 2000 ./data/book/books3-train.jsonl ./data/book/split/books3_
split -a 3 -d -l 2000 ./data/book/pg19-train.jsonl ./data/book/split/pg19_
cat ./data/book/pg19-test.jsonl > ./data/book/split/pg19_014
cat ./data/book/pg19-validation.jsonl >> ./data/book/split/pg19_014
python ./book/dedup.py -w W -k K -l L -n N
```

After downloading the dataset, run the following command to count tokens. (Assuming you are in `data_prep`)

```bash
rm ./data/book/books3-train.jsonl
rm ./data/book/pg19-*
python ./book/token_count.py
```

## License


The file [dedup.py](dedup.py) was co-developed with [Ontocord.ai](https://www.ontocord.ai).

```
Copyright 2023 Ontocord.ai, Together Computer, ETH Zürich, Stanford University

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

For full terms, see the LICENSE file at the root of this repo. If you have any questions, comments, or concerns about licensing please [contact us](https://www.together.xyz/contact).
