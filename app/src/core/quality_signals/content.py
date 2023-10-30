import re
import sys
import operator
from pathlib import Path
from typing import List, Tuple, Type

from core.constants import PRECISION
from core.quality_signals.base import RPSBase
from core.quality_signals.utils.stop_words import get_stop_words
from core.document import Document
from core.data_types import SignalType
from core.quality_signals.utils.content import \
    load_bad_words, load_bad_urls_index
from utilities.register.registry_utils import *
from utilities.text import form_ngrams

__all__ = ["register_content_callables", "content_schema"]


def content_schema() -> List[Tuple[str, Type]]:
    r""" Returns a list of signal names and their data types """
    return signal_schema(module=sys.modules[__name__])


def register_content_callables(
        language: str, bad_urls_dir: str, bad_words_dir: str
) -> List[RPSBase]:
    r""" Returns a list of signal functions (i.e., RPSBase instances) that
    are used to extract content signals from a document.

    Args:
        language: The language of the document.
        bad_urls_dir: directory containing the UT1 blacklist.
        bad_words_dir: directory containing the LDNOOBW blacklist.

    Returns:
        A list of signal function class instances.
    """
    return list(map(
        lambda cls: cls(
            language=language,
            bad_urls_dir=bad_urls_dir,
            bad_words_dir=bad_words_dir
        ),
        get_callables_from_module(module=sys.modules[__name__])
    ))


class RPS_Doc_LDNOOBW_Words(RPSBase):  # noqa
    r""" The number of sequences of words that are contained in the
    List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words blocklist. The
    blocklist is obtained from
    https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
    """
    __slots__ = ["_block_words", "_gram_vals"]

    def __init__(
            self, bad_words_dir: str, language: str, *args, **kwargs  # noqa
    ):
        super(RPS_Doc_LDNOOBW_Words, self).__init__()
        self._block_words = load_bad_words(
            bad_words_dir=Path(bad_words_dir), lang=language
        )

        # cache the number of words in each block list entry
        self._gram_vals = set(map(
            lambda w: 1 + operator.countOf(w, " "), self._block_words
        ))

    def __call__(self, document: Document) -> SignalType:
        if len(document.normalized_content) == 0:
            return [(0, len(document), .0)]

        num_dirty = 0

        # for each ngram value, count the number of ngrams in the document
        # which are also in the block words list
        for n in self._gram_vals:
            if n == 1:
                num_dirty += sum(
                    1 for _ in filter(
                        lambda w: w in self._block_words,
                        document.normalized_words
                    )
                )
                continue

            num_dirty += sum(
                1 for _ in filter(
                    lambda t: " ".join(t) in self._block_words,
                    # try to fetch the cached ngrams, otherwise compute them
                    # on the fly
                    getattr(document, f"norm_{n}grams", None)
                    or
                    form_ngrams(iter(document.normalized_words), n)
                )
            )

        score = float(num_dirty)
        return [(0, len(document), score)]


class RPS_Doc_Lorem_Ipsum(RPSBase):  # noqa
    r""" The ratio between the number of occurences of 'lorem ipsum'
    and the number of characters in the text after normalization. Text is
    normalized by lowercasing and removing punctuation. """
    SEARCH_TEXT = "lorem ipsum"
    SEARCH_REGEX = re.compile(r"lorem ipsum", re.IGNORECASE)

    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        if len(document.normalized_content) == 0:
            return [(0, len(document), 0.0)]

        if self.SEARCH_TEXT not in document.normalized_content:
            return [(0, len(document), .0)]

        num_occurences = len(self.SEARCH_REGEX.findall(
            document.normalized_content
        ))

        score = float(num_occurences) / len(document.normalized_content)
        score = round(score, PRECISION)

        return [(0, len(document), score)]


class RPS_Doc_Curly_Bracket(RPSBase):  # noqa
    r""" The ratio between the number of occurences of '{' or '}' and the
    number of characters in the raw text. """
    SEARCH_TEXT = ("{", "}")
    __slots__ = ()

    def __call__(self, document: Document) -> SignalType:
        if len(document.raw_content) == 0:
            return [(0, len(document), .0)]

        if all(map(lambda x: x not in document.raw_content, self.SEARCH_TEXT)):
            return [(0, len(document), .0)]

        num_occurences = sum(
            map(lambda x: operator.countOf(document.raw_content, x),
                self.SEARCH_TEXT)
        )

        score = float(num_occurences) / len(document.raw_content)
        score = round(score, PRECISION)

        return [(0, len(document), score)]


class RPS_Doc_UT1_Blacklist(RPSBase):  # noqa
    r""" An categorical id of the list of categories of the domain of the
    document. Categories are obtained from the UT1 blacklist.
    """
    __slots__ = ["_ut1_mapping"]

    def __init__(self, bad_urls_dir: str, *args, **kwargs):  # noqa
        super(RPS_Doc_UT1_Blacklist, self).__init__()
        self._ut1_mapping = load_bad_urls_index(Path(bad_urls_dir))

    def __call__(self, document: Document) -> SignalType:
        score: int = self._ut1_mapping.get(document.domain, None)
        return [(0, len(document), score)]


class RPS_Doc_Stop_Word_Fraction(RPSBase):  # noqa
    r""" The ratio between the number of stop words and the number of words in
    the document. """
    __slots__ = ["_stop_words"]

    def __init__(self, language: str, *args, **kwargs):  # noqa
        super(RPS_Doc_Stop_Word_Fraction, self).__init__()
        self._stop_words = get_stop_words(language)

    def __call__(self, document: Document) -> SignalType:
        if len(document.normalized_words) == 0:
            return [(0, len(document), .0)]

        num_stop_words = sum(
            map(lambda w: w in self._stop_words, document.raw_words)
        )

        score = float(num_stop_words) / document.num_raw_words
        score = round(score, PRECISION)

        return [(0, len(document), score)]
