from nltk.tokenize import WordPunctTokenizer
import re
from typing import Optional, Tuple, Callable

from utilities.text import normalize, form_ngrams
from core.data_types import TextSlice
from core.quality_signals.utils.dsir import hash_feature

_word_tokenizer = WordPunctTokenizer()


def _compute_ngrams(text_seq, n):
    return tuple(form_ngrams(iter(text_seq), n))


def split_paragraphs(
        text: str, normalizer: Callable[[str], str], remove_empty: bool = True
) -> Tuple[TextSlice]:
    """
    This function is adapted from dolma: https://github.com/allenai/dolma

    Split a string into paragraphs. A paragraph is defined as a sequence of
    zero or more characters, followed by a newline character, or a sequence
     of one or more characters, followed by the end of the string.
    """
    text_slices = tuple(
        TextSlice(normalizer(text[match.start():match.end()]), match.start(),
                  match.end())
        for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text)
    )

    if remove_empty is True:
        text_slices = tuple(
            text_slice for text_slice in text_slices if text_slice[0].strip()
        )

    return text_slices


class Document:
    __slots__ = (
        "_raw_content", "_normalized_content", "_raw_lines",
        "_normalized_lines", "_raw_words", "_normalized_words",
        "_num_raw_words", "_num_normalized_words", "_domain", "_raw_2grams",
        "_raw_3grams", "_norm_2grams", "_norm_3grams", "_norm_4grams",
        "_hash_features"
    )

    def __init__(
            self, content: str, domain: Optional[str],
            precompute_ngrams: bool = False,
            precompute_hash_features: bool = False,
            dsir_buckets: Optional[int] = None
    ):
        self._raw_content = content
        self._domain = domain

        # the normalized content: lowercased and punctuation removed
        self._normalized_content = normalize(content)

        # the lines of the document (split by newline)
        self._raw_lines: Tuple[TextSlice] = split_paragraphs(
            text=content, normalizer=lambda x: x, remove_empty=False
        )

        # the lines of the document (split by newline), normalized
        self._normalized_lines: Tuple[TextSlice] = split_paragraphs(
            text=content, normalizer=normalize, remove_empty=False
        )

        # the words of the document after normalization
        self._raw_words = tuple(_word_tokenizer.tokenize(self._raw_content))

        # the normalized words of the document (split by whitespace)
        self._normalized_words = tuple(self._normalized_content.split())

        # get number of words before and after normalization
        self._num_raw_words = len(self._raw_words)
        self._num_normalized_words = len(self._normalized_words)

        # precompute ngrams
        if precompute_ngrams:
            # raw grams
            self._raw_2grams = _compute_ngrams(self._raw_words, 2)
            self._raw_3grams = _compute_ngrams(self._raw_words, 3)

            # normalized grams
            self._norm_2grams = _compute_ngrams(self._normalized_words, 2)
            self._norm_3grams = _compute_ngrams(self._normalized_words, 3)
            self._norm_4grams = _compute_ngrams(self._normalized_words, 4)
        else:
            self._raw_2grams = None
            self._raw_3grams = None
            self._norm_2grams = None
            self._norm_3grams = None
            self._norm_4grams = None

        # precomupte hash features
        if precompute_hash_features:
            bigrams = self._raw_2grams or _compute_ngrams(self._raw_words, 2)
            self._hash_features = hash_feature(
                unigrams=self._raw_words,
                bigrams=bigrams,
                buckets=dsir_buckets
            )
        else:
            self._hash_features = None

    def __len__(self):
        return len(self._raw_content)

    @property
    def raw_content(self):
        return self._raw_content

    @property
    def normalized_content(self):
        return self._normalized_content

    @property
    def raw_lines(self):
        return self._raw_lines

    @property
    def normalized_lines(self):
        return self._normalized_lines

    @property
    def raw_words(self):
        return self._raw_words

    @property
    def normalized_words(self):
        return self._normalized_words

    @property
    def num_raw_words(self):
        return self._num_raw_words

    @property
    def num_normalized_words(self):
        return self._num_normalized_words

    @property
    def domain(self):
        return self._domain

    @property
    def raw_1grams(self):
        return self._raw_words

    @property
    def raw_2grams(self):
        return self._raw_2grams

    @property
    def raw_3grams(self):
        return self._raw_3grams

    @property
    def norm_1grams(self):
        return self._normalized_words

    @property
    def norm_2grams(self):
        return self._norm_2grams

    @property
    def norm_3grams(self):
        return self._norm_3grams

    @property
    def norm_4grams(self):
        return self._norm_4grams

    @property
    def hash_features(self):
        return self._hash_features
