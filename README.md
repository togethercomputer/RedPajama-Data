<a href="https://discord.gg/9Rk6sSeWEG"><img src="https://img.shields.io/discord/1082503318624022589?label=discord" /></a> <img src="https://img.shields.io/github/license/togethercomputer/RedPajama-Data" />

# RedPajama-Data: An Open Source Recipe to Reproduce LLaMA training dataset

<img width="500" src="docs/redpajama.png" />

This repo contains a reproducible data receipe for the RedPajama data, with the following token counts:

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


## Data Preparation

In `data_prep`, we provide all pre-processing scripts and guidelines.

## Tokenization

In `tokenization`, we provide an example of how to tokenize the dataset using the GPT-NeoX tokenizer.

## Visualization

In `viz`, we provide a dashboard for exploring a subset of the data using [Meerkat](https://github.com/hazyresearch/meerkat).

## License

The code in this repo is licensed under the Apache 2.0 license. Unless otherwise noted,

```
Copyright 2023 Together Computer, ETH Zürich, Stanford University

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

The file [data\_prep/book/dedup.py](data_prep/book/dedup.py) was co-developed with [Ontocord.ai](https://www.ontocord.ai).

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

The dataset itself, please refer to the licenses of the data subsets you use.

* [Common Crawl Foundation Terms of Use](https://commoncrawl.org/terms-of-use/full/)
* [C4 license](https://huggingface.co/datasets/allenai/c4#license)
* GitHub was limited to MIT, BSD, or Apache licenses only
* Books: [the\_pile\_books3 license](https://huggingface.co/datasets/the_pile_books3#licensing-information) and [pg19 license](https://huggingface.co/datasets/pg19#licensing-information)
* [ArXiv Terms of Use](https://info.arxiv.org/help/api/tou.html)
* [Wikipedia License](https://huggingface.co/datasets/wikipedia#licensing-information)
* [StackExchange license on the Internet Archive](https://archive.org/details/stackexchange)

For full terms, see the LICENSE file. If you have any questions, comments, or concerns about licensing please [contact us](https://www.together.xyz/contact).

## Acknowledgement

We are appreciative to the work done by the growing open-source AI community that made this project possible. That includes:
- Participants in building the RedPajama dataset including [Ontocord.ai](Ontocord.ai), [MILA Québec AI Institute](https://mila.quebec/en/), [ETH DS3Lab](https://ds3lab.inf.ethz.ch/), [Université de Montréal](https://www.umontreal.ca/), [Stanford Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/), [Stanford Hazy Research research group](https://hazyresearch.stanford.edu/) and [LAION](https://laion.ai/).  
- [EleutherAI](https://www.eleuther.ai/) — This project is built on the backs of the great team at EleutherAI — including the source code they provided for training GPT-NeoX. 
- An award of computer time was provided by the [INCITE program](https://www.alcf.anl.gov/science/incite-allocation-program). This research also used resources of the [Oak Ridge Leadership Computing Facility (OLCF)](https://www.together.xyz/blog/redpajama#:~:text=resources%20of%20the-,Oak%20Ridge%20Leadership%20Computing%20Facility%20(OLCF),-%2C%20which%20is%20a), which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725.


