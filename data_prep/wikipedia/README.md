### Wikipedia

Follow these instructions to create the Wikipedia dataset.

We use the Wikipedia dataset available on Huggingface, which is based on Wikipedia dumps from 2023-03-20 and contains
text in 20 different languages.

Run these commands to install the parser to remove hyperlinks and templates in Wikipedia pages.

```
pip install apache_beam mwparserfromhell
```

If the installed version mwparserfromhell is lower than 0.7, follow the following link to install the latest
mwparserfromhell. Otherwise, there may occur errors when processing the Spanish Wikipedia dump. (
See https://github.com/huggingface/datasets/issues/577)

```
https://github.com/earwig/mwparserfromhell
```

To download the dataset, run the following commands. (Assuming you are in `data_prep`)

```bash
mkdir -p data/wikipedia
python ./wikipedia/download.py
python ./wikipedia/convert_format.py
```

After downloading the dataset, run the following command to count tokens.

```bash
python ./wikipedia/token_count.py
```