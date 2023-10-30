import re
import string
import unicodedata

TRANSLATION_TABLE_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(
        text: str,
        remove_punct: bool = True,
        lowercase: bool = True,
        nfd_unicode: bool = True,
        white_space: bool = True
) -> str:
    """ Normalize the text by lowercasing and removing punctuation. """
    # remove punctuation
    if remove_punct:
        text = text.translate(TRANSLATION_TABLE_PUNCTUATION)

    # lowercase
    if lowercase:
        text = text.lower()

    if white_space:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

    # NFD unicode normalization
    if nfd_unicode:
        text = unicodedata.normalize("NFD", text)

    return text
