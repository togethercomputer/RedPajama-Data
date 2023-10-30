import random
import string

# text normalization: lowercasing and removing punctuation
TRANSLATION_TABLE = str.maketrans(
    string.ascii_lowercase + string.ascii_uppercase + "\n",
    string.ascii_lowercase * 2 + " ",
    string.punctuation
)


def normalize_text(text: str, max_words: int = -1):
    r""" Normalize text by lowercasing and removing punctuation; if max words
    is larger than 0, then a random but contiguous span of max_words is
    selected from the text.

    Args:
        text: text to normalize
        max_words: maximum number of words to keep in text

    Returns:
        normalized text
    """
    text = text.translate(TRANSLATION_TABLE)
    text = text.split()
    num_words = len(text)

    # pick a random span inside the text if it is too long
    if num_words > max_words > 0:
        start = random.randint(0, num_words - max_words)
        text = text[start:start + max_words]

    return " ".join(text)
