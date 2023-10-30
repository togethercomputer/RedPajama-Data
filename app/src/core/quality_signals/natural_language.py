from collections import Counter
import math
import re
import sys
from typing import List, Tuple, Type

from core.constants import PRECISION
from core.data_types import SignalType
from core.quality_signals.base import RPSBase
from core.document import Document
from utilities.register.registry_utils import *

__all__ = [
    "register_natural_language_callables",
    "natural_language_schema"
]


def natural_language_schema() -> List[Tuple[str, Type]]:
    r""" Returns a list of signal names and their data types """
    return signal_schema(module=sys.modules[__name__])


def register_natural_language_callables() -> List[RPSBase]:
    r""" Returns a list of signal functions (i.e., RPSBase instances) that
    are used to extract natural language signals from a document.

    Returns:
        A list of signal function class instances.
    """
    return list(map(
        lambda cls: cls(),
        get_callables_from_module(module=sys.modules[__name__])
    ))


class RPS_Doc_Num_Sentences(RPSBase):  # noqa
    r""" The number of sentences in the content. This is calculated using
    the regex r'\b[^.!?]+[.!?]*' """
    SENT_PATTERN = re.compile(r'\b[^.!?]+[.!?]*', flags=re.UNICODE)

    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        r""" count the number of sentences in the content using regex"""
        score = float(len(self.SENT_PATTERN.findall(document.raw_content)))
        return [(0, len(document), score)]


class RPS_Doc_Word_Count(RPSBase):  # noqa
    r""" The number of words in the content after normalization. """
    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        return [(0, len(document), document.num_normalized_words)]


class RPS_Doc_Mean_Word_Length(RPSBase):  # noqa
    r""" The mean length of words in the content normalization. """
    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        if document.num_normalized_words == 0:
            return [(0, len(document), None)]

        num_chars = float(sum(map(len, document.normalized_words)))
        score = num_chars / document.num_normalized_words
        score = round(score, PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_Symbol_To_Word_Ratio(RPSBase):  # noqa
    r""" The ratio of symbols to words in the content. This is analogous to
    the signal used in Gopher. Symbols are defined "#", "...", and "…". """
    SYMBOLS = ("#", "...", "…")

    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        num_words = document.num_raw_words

        if num_words == 0:
            return [(0, len(document), None)]

        # count the number of symbols in the content
        num_symbols = float(sum(
            document.raw_content.count(x) for x in self.SYMBOLS
        ))

        score = num_symbols / num_words
        score = round(score, PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_Frac_Lines_End_With_Ellipsis(RPSBase):  # noqa
    r""" The fraction of lines that end with an ellipsis, where an ellipsis
    is defined as either "..." or "…". """
    ELLIPSIS_SYMBOLS = ("...", "…")

    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        num_lines = len(document.raw_lines)

        if num_lines == 0:
            return [(0, len(document), None)]

        total_ellipsis_lines = float(sum(
            text_slice.text.rstrip().endswith(self.ELLIPSIS_SYMBOLS)
            for text_slice in document.raw_lines
        ))

        score = total_ellipsis_lines / num_lines
        score = round(score, PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_Frac_No_Alph_Words(RPSBase):  # noqa
    r""" The fraction of words that contain no alphabetical character.
    This is based on the raw content. """
    ALPH_REGEX = re.compile(r"[a-zA-Z]")

    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        num_words = document.num_raw_words

        if num_words == 0:
            return [(0, len(document), None)]

        num_words_with_alpha = float(sum(
            int(self.ALPH_REGEX.search(word) is not None)
            for word in document.raw_words
        ))

        score = 1.0 - num_words_with_alpha / num_words
        score = round(score, PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_Frac_Unique_Words(RPSBase):  # noqa
    r""" The fraction of unique words in the content. This is also known as
    the degeneracy of a text sample. Calculated based on the normalized
    content. """
    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        num_words = document.num_normalized_words

        if num_words == 0:
            return [(0, len(document), None)]

        score = float(len(set(document.normalized_words))) / num_words
        score = round(score, PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_Unigram_Entropy(RPSBase):  # noqa
    r""" The entropy of the unigram distribution of the
    content. This measures the diversity of the content and is computed
    using sum(-x / total * log(x / total)) where the sum is taken over
    over counts of unique words in the noramlized (punctuation removed,
    lowercased) content."""
    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        if len(document.normalized_words) == 0:
            return [(0, len(document), None)]

        # count the number of times each word appears in the content
        counter = Counter(document.normalized_words)

        # calculate the entropy of the unigram distribution
        total = sum(counter.values())
        entropy = sum(map(
            lambda x: -x / total * math.log(x / total) if x > 0 else 0.0,
            counter.values()
        ))

        score = round(entropy, PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_Frac_All_Caps_Words(RPSBase):  # noqa
    r""" The fraction of words in the content that only conist of uppercase
    letters. This is based on the raw content."""
    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        num_words = document.num_raw_words

        if num_words == 0:
            return [(0, len(document), None)]

        score = float(sum(map(str.isupper, document.raw_words))) / num_words
        score = round(score, PRECISION)
        return [(0, len(document), score)]
