import sys
from typing import List, Tuple, Type

from core.constants import PRECISION
from core.quality_signals.base import RPSBase
from core.data_types import SignalType, ScoreType, TextSlice
from core.document import Document
from utilities.register.registry_utils import *

__all__ = [
    "register_lines_callables", "lines_schema"
]


def lines_schema() -> List[Tuple[str, Type]]:
    r""" Returns a list of signal names and their data types """
    return signal_schema(module=sys.modules[__name__])


def register_lines_callables() -> List[RPSBase]:
    r""" Returns a list of signal functions (i.e., RPSBase instances) that
    are used to extract line signals from a document.

    Returns:
        A list of signal function class instances.
    """
    return list(map(
        lambda cls: cls(),
        get_callables_from_module(module=sys.modules[__name__])
    ))


class RPS_Lines_Javascript_Counts(RPSBase):  # noqa
    r""" The number of occurences of the word "javascript" in each line. """
    SEARCH_TEXT = "javascript"
    __slots__ = ()

    def _process_line(self, text_slice: TextSlice) -> ScoreType:
        if len(text_slice.text) == 0:
            return tuple((text_slice.start, text_slice.end, 0.0))

        score = float(sum(
            1 for w in text_slice.text.split() if w == self.SEARCH_TEXT
        ))

        return tuple((text_slice.start, text_slice.end, score))

    def __call__(self, document: Document) -> SignalType:
        return list(map(self._process_line, document.normalized_lines))


class RPS_Lines_Ending_With_Terminal_Punctution_Mark(RPSBase):  # noqa
    r""" A list of integers indicating whether (1) or not (0) a line ends with
    a terminal punctuation mark. A terminal punctation mark is defined as
    one of the following: ".", "!", "?", "”" """
    TERMINAL_PUNCTUATION_MARKS = (".", "!", "?", "”")
    __slots__ = ()

    def _process_line(self, text_slice: TextSlice) -> ScoreType:
        score = text_slice.text.rstrip().endswith(
            self.TERMINAL_PUNCTUATION_MARKS
        )
        score = float(score)
        return tuple((text_slice.start, text_slice.end, score))

    def __call__(self, document: Document) -> SignalType:
        return list(map(self._process_line, document.raw_lines))


class RPS_Lines_Num_Words(RPSBase):  # noqa
    r""" The number of words in each line. This is computed based on the
    normalied text. Normalization is done by lowercasing the text and
    removing punctuation."""
    __slots__ = ()

    def _process_line(self, text_slice: TextSlice) -> ScoreType:  # noqa
        score = len(text_slice.text.split())
        return tuple((text_slice.start, text_slice.end, score))

    def __call__(self, document: Document) -> SignalType:
        return list(map(self._process_line, document.normalized_lines))


class RPS_Lines_Uppercase_Letter_Fraction(RPSBase):  # noqa
    r""" The ratio between number of uppercase letters and total number of
    characters in each line. This is based on the raw text. """
    __slots__ = ()

    def _process_line(self, text_slice: TextSlice) -> ScoreType:  # noqa
        if len(text_slice) == 0:
            return tuple((text_slice.start, text_slice.end, 0.0))

        score = sum(map(str.isupper, text_slice.text)) / len(text_slice)
        score = round(score, PRECISION)
        return tuple((text_slice.start, text_slice.end, score))

    def __call__(self, document: Document) -> SignalType:
        return list(map(self._process_line, document.raw_lines))


class RPS_Lines_Numerical_Chars_Fraction(RPSBase):  # noqa
    r""" The ratio between number of numerical characters and total number of
    characters in each line. This is based on text after lowercasing and
    removing punctuation."""
    __slots__ = ()

    def _process_line(self, text_slice: TextSlice) -> ScoreType:  # noqa
        if len(text_slice) == 0:
            return tuple((text_slice.start, text_slice.end, 0.0))

        score = sum(map(str.isnumeric, text_slice.text)) / len(text_slice)
        score = round(score, PRECISION)
        return tuple((text_slice.start, text_slice.end, score))

    def __call__(self, document: Document) -> SignalType:
        return list(map(self._process_line, document.normalized_lines))


class RPS_Lines_Start_With_Bulletpoint(RPSBase):  # noqa
    r""" Whether the lines that start with a bullet point symbol. The
    following set of unicodes are considered a bullet point:
    \u2022 (bullet point), \u2023 (triangular bullet point), \u25B6 (black
    right pointing triangle), \u25C0 (black left pointing triangle),
    \u25E6 (white bullet point), \u25A0 (black square), \u25A1 (white
    square), \u25AA (black small square), \u25AB (white small square),
    \u2013 (en dash)."""
    BULLET_POINT_SYMBOLS = (
        "\u2022",  # bullet point
        "\u2023",  # triangular bullet point
        "\u25B6",  # black right pointing triangle
        "\u25C0",  # black left pointing triangle
        "\u25E6",  # white bullet point
        "\u25A0",  # black square
        "\u25A1",  # white square
        "\u25AA",  # black small square
        "\u25AB",  # white small square
        "\u2013",  # en dash
    )

    __slots__ = ()

    def _process_line(self, text_slice: TextSlice) -> ScoreType:  # noqa
        score = text_slice.text.lstrip().startswith(self.BULLET_POINT_SYMBOLS)
        score = float(score)
        return tuple((text_slice.start, text_slice.end, score))

    def __call__(self, document: Document) -> SignalType:
        num_lines = len(document.raw_lines)

        if num_lines == 0:
            return [(0, len(document), None)]

        return list(map(self._process_line, document.raw_lines))
