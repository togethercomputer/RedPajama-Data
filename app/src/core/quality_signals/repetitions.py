from collections import Counter
import numpy as np
import sys
from typing import List, Tuple, Type

from core.constants import PRECISION
from core.quality_signals.base import RPSBase
from core.document import Document
from core.data_types import SignalType
from utilities.register.registry_utils import *
from utilities.text import form_ngrams

__all__ = [
    "register_repetitions_callables",
    "repetitions_schema"
]


def repetitions_schema() -> List[Tuple[str, Type]]:
    r""" Returns a list of signal names and their data types """
    return signal_schema(module=sys.modules[__name__])


def register_repetitions_callables() -> List[RPSBase]:
    r""" Returns a list of signal functions (i.e., RPSBase instances) that
    are used to extract repetition related signals from a document.

    Returns:
        A list of signal function class instances.
    """
    return list(map(
        lambda cls: cls(),
        get_callables_from_module(module=sys.modules[__name__])
    ))


class Base_RPS_Frac_Chars_In_Top_NGram(RPSBase):  # noqa
    r""" Base class for calculating the fraction of characters in the
    top N-gram. This operates on the lower-cased, punctation removed
    content."""
    NGRAM_SIZE: int = None

    __slots__ = []

    def __call__(self, document: Document) -> SignalType:
        if self.NGRAM_SIZE is None:
            raise NotImplementedError(
                "NGRAM_SIZE must be set in the subclass"
            )

        # get the most common ngram
        most_common_ngram = Counter(
            # fetch the ngrams from the document if they exist, otherwise
            # compute them
            getattr(document, f"norm_{self.NGRAM_SIZE}grams", None)
            or
            form_ngrams(iter(document.normalized_words), self.NGRAM_SIZE)
        ).most_common(1)

        if len(most_common_ngram) == 0:
            return [(0, len(document), 0.0)]

        ngram, count = most_common_ngram[0]

        if count <= 1:
            return [(0, len(document), 0.0)]

        total_chars = sum(len(w) for w in document.normalized_words)
        score = sum(len(w) for w in ngram) * count / total_chars
        score = round(score, PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_Frac_Chars_Top_2gram(Base_RPS_Frac_Chars_In_Top_NGram):  # noqa
    r""" The fraction of characters in the top word Bigram. Operates on the
    lower-cased, punctation removed content."""
    NGRAM_SIZE = 2
    __slots__ = []


class RPS_Doc_Frac_Chars_Top_3gram(Base_RPS_Frac_Chars_In_Top_NGram):  # noqa
    r""" The fraction of characters in the top word Trigram. Operates on the
    lower-cased, punctation removed content."""
    NGRAM_SIZE = 3
    __slots__ = []


class RPS_Doc_Frac_Chars_Top_4gram(Base_RPS_Frac_Chars_In_Top_NGram):  # noqa
    r""" The fraction of characters in the top word 4gram. Operates on the
    lower-cased, punctation removed content."""
    NGRAM_SIZE = 4
    __slots__ = []


class Base_RPS_Frac_Chars_In_Dupe_NGrams(RPSBase):  # noqa
    r""" Base class for calculating the fraction of characters in
    duplicate word N-grams. This operates on the lower-cased, punctation
    removed content. The function also ensures that characters in overlapping
    ngrams are only counted once."""
    NGRAM_SIZE: int = None
    __slots__ = []

    def __call__(self, document: Document) -> SignalType:
        if self.NGRAM_SIZE is None:
            raise NotImplementedError(
                "NGRAM_SIZE must be set in the subclass"
            )

        if len(document.normalized_words) < self.NGRAM_SIZE:
            return [(0, len(document), 0.0)]

        # fetch the ngrams from the document if they exist, otherwise
        # compute them
        doc_n_grams = (
                getattr(document, f"norm_{self.NGRAM_SIZE}grams", None)
                or
                tuple(form_ngrams(
                    iter(document.normalized_words), self.NGRAM_SIZE
                ))
        )

        # keep only ngrams which occur at least twice
        ngram_dupes = {
            ngram for ngram, count in Counter(doc_n_grams).items() if count > 1
        }

        duplicated_grams = np.zeros(len(document.normalized_words), dtype=int)

        i = 0
        for ngram in doc_n_grams:
            if ngram in ngram_dupes:
                duplicated_grams[i: i + self.NGRAM_SIZE] = 1

            i += 1

        word_lengths = np.array(list(map(len, document.normalized_words)))
        chars_duped = np.sum(word_lengths * duplicated_grams)
        total_chars = np.sum(word_lengths)

        if total_chars == 0:
            return [(0, len(document), 0.0)]

        score = float(chars_duped / total_chars)
        score = round(score, PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_Frac_Chars_Dupe_5Grams(  # noqa
    Base_RPS_Frac_Chars_In_Dupe_NGrams
):
    r""" The fraction of characters in duplicate word 5grams. This operates on
    the lower-cased, punctation removed content. It is also ensured that
    characters in overlapping ngrams are only counted once. """
    NGRAM_SIZE = 5
    __slots__ = []


class RPS_Doc_Frac_Chars_Dupe_6Grams(  # noqa
    Base_RPS_Frac_Chars_In_Dupe_NGrams
):
    r""" The fraction of characters in duplicate word 6grams. This operates on
    the lower-cased, punctation removed content. It is also ensured that
    characters in overlapping ngrams are only counted once. """
    NGRAM_SIZE = 6
    __slots__ = []


class RPS_Doc_Frac_Chars_Dupe_7Grams(  # noqa
    Base_RPS_Frac_Chars_In_Dupe_NGrams
):
    r""" The fraction of characters in duplicate word 7grams. This operates on
    the lower-cased, punctation removed content. It is also ensured that
    characters in overlapping ngrams are only counted once. """
    NGRAM_SIZE = 7
    __slots__ = []


class RPS_Doc_Frac_Chars_Dupe_8Grams(  # noqa
    Base_RPS_Frac_Chars_In_Dupe_NGrams
):
    r""" The fraction of characters in duplicate word 8grams. This operates on
    the lower-cased, punctation removed content. It is also ensured that
    characters in overlapping ngrams are only counted once. """
    NGRAM_SIZE = 8
    __slots__ = []


class RPS_Doc_Frac_Chars_Dupe_9Grams(  # noqa
    Base_RPS_Frac_Chars_In_Dupe_NGrams
):
    r""" The fraction of characters in duplicate word 9grams. This operates on
    the lower-cased, punctation removed content. It is also ensured that
    characters in overlapping ngrams are only counted once. """
    NGRAM_SIZE = 9
    __slots__ = []


class RPS_Doc_Frac_Chars_Dupe_10Grams(  # noqa
    Base_RPS_Frac_Chars_In_Dupe_NGrams
):
    r""" The fraction of characters in duplicate word 10grams. This operates on
    the lower-cased, punctation removed content. It is also ensured that
    characters in overlapping ngrams are only counted once. """
    NGRAM_SIZE = 10
    __slots__ = []
