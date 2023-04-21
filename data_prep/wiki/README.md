### Wikipedia

Follow these instructions to create the Wikipedia dataset.

We use the Wikipedia dataset available on Huggingface, which is based on Wikipedia dumps from 2023-03-20 and contains
text in 20 different languages.

#### Setup

As a preprocessing step, hyperlinks and templates are removed from wikipedia pages. This requires apache_beam and
mwparserfromhell to be installed. To install the former, run:

```bash
pip install apache_beam
```

The `mwparserfromhell` library needs to be version >= 0.7 to process the spanish Wikipedia dump.
This is currently only available as a dev version (see the corresponding
[github issue](https://github.com/huggingface/datasets/issues/577.) for reference). To install it, run:

```bash
pip install git+https://github.com/earwig/mwparserfromhell.git@0f89f44
```

Finally, you need to install version `0.3.5.1` of dill:

```bash
pip install dill==0.3.5.1
```

#### Downloading the dataset

To download and prepare the dataset, run the following commands from `data_prep`:

```bash
mkdir -p data/wikipedia
python ./wiki/download.py --data_dir data/wikipedia
python ./wiki/convert_format.py
```

After downloading the dataset, run the following command to count tokens.

```bash
python ./wiki/token_count.py
```