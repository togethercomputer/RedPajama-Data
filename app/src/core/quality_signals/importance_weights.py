import numpy as np
import scipy.stats as stats
import sys
from typing import List, Tuple, Type, Optional
from pathlib import Path

from core.constants import PRECISION
from core.quality_signals.base import RPSBase
from core.quality_signals.utils.dsir import hash_feature
from core.document import Document
from core.data_types import SignalType

from utilities.register.registry_utils import *
from utilities.text import form_ngrams

__all__ = [
    "register_importance_weights_callables",
    "importance_weights_schema"
]


def importance_weights_schema() -> List[Tuple[str, Type]]:
    r""" Returns a list of signal names and their data types """
    return signal_schema(module=sys.modules[__name__])


def register_importance_weights_callables(
        source_fps: Optional[Tuple[str]],
        wiki_fps: Optional[Tuple[str]],
        openwebtext_fps: Optional[Tuple[str]],
        books_fps: Optional[Tuple[str]],
        language: str
) -> List[RPSBase]:
    r""" Returns a list of signal functions (i.e., RPSBase instances) that
    are used to extract content signals from a document.

    Returns:
        A list of signal function class instances.
    """
    return list(map(
        lambda cls: cls(
            language=language,
            source_fps=source_fps,
            wiki_fps=wiki_fps,
            openwebtext_fps=openwebtext_fps,
            books_fps=books_fps
        ),
        get_callables_from_module(module=sys.modules[__name__])
    ))


class Base_Importance(RPSBase):  # noqa
    r""" Base class for functions which return the log ratio of the likelihood
    of the document's features with respect to the target domain
    versus the source domain. """

    __slots__ = (
        "_log_diff_dist", "_feature_dim", "_target_lambda",
        "_source_lambda", "_length_correction"
    )

    def __init__(
            self,
            target_fps: Tuple[str, str],
            source_fps: Tuple[str, str],
            language: str,
            length_correction: bool = False
    ):
        super(Base_Importance, self).__init__()
        self._length_correction = length_correction

        if target_fps is None or source_fps is None:
            self._log_diff_dist = None
            self._feature_dim = None
            return

        target_count_fp, target_lambbda_fp = target_fps
        source_count_fp, source_lambda_fp = source_fps

        assert language == Path(target_count_fp).stem.split(".")[1], \
            f"Language mismatch between {target_count_fp} and {language}"

        assert language == Path(source_count_fp).stem.split(".")[1], \
            f"Language mismatch between {target_count_fp} and {language}"

        # load hash counts
        target_counts = np.load(target_count_fp)
        target_dist = target_counts / target_counts.sum()
        source_counts = np.load(source_count_fp)
        source_dist = source_counts / source_counts.sum()

        if length_correction:
            self._target_lambda = np.load(target_lambbda_fp)
            self._source_lambda = np.load(source_lambda_fp)
        else:
            self._target_lambda = None
            self._source_lambda = None

        # compute log diff dist
        self._feature_dim = target_counts.shape[0]
        self._log_diff_dist = np.array(
            np.log(target_dist + 1e-8) - np.log(source_dist + 1e-8)
        )

    def __call__(self, document: Document) -> SignalType:
        if self._log_diff_dist is None:
            return [(0, len(document), None)]

        doc_len = len(document)

        if doc_len == 0:
            return [(0, doc_len, None)]

        # try to fetch cached features, if not compute them
        features = (
            document.hash_features
            if document.hash_features is not None
            else
            hash_feature(
                unigrams=document.raw_words,
                # fetch cached bigrams, otherwise comptue them
                bigrams=(
                        document.raw_2grams
                        or
                        tuple(form_ngrams(iter(document.raw_words), 2))
                ),
                buckets=self._feature_dim
            )
        )

        logratio = np.inner(features, self._log_diff_dist)
        score = float(logratio)

        if not self._length_correction:
            score = round(score, PRECISION)
            return [(0, doc_len, score)]

        # correct for the length assuming a Poisson distribution
        return self.__add_length_penalty(score, doc_len)

    def __add_length_penalty(self, score, doc_len):
        # correct for the length assuming a Poisson distribution
        len_prob_source = stats.poisson.pmf(doc_len, self._source_lambda)
        len_prob_target = stats.poisson.pmf(doc_len, self._target_lambda)

        len_correction = np.log(len_prob_target + 1e-8) - \
                         np.log(len_prob_source + 1e-8)

        score += float(len_correction)
        score = round(score, PRECISION)
        return [(0, doc_len, score)]


class RPS_Doc_Wikipedia_Importance(Base_Importance):  # noqa
    r""" Given a bag of {1,2}-wordgram model trained on Wikipedia articles p,
    and a model trained on the source domain q. This is the logarithm of the
    ratio p(doc)/q(doc). If length_correction is enabled, then the length of
    score is adjusted by adding the term log(p_poisson(len) / q_poisson(len))
    to the final score.
    """
    __slots__ = ()

    def __init__(
            self,
            wiki_fps: Tuple[str, str],
            source_fps: Tuple[str, str],
            language: str,
            *args, **kwargs  # noqa
    ):
        super(RPS_Doc_Wikipedia_Importance, self).__init__(
            target_fps=wiki_fps,
            source_fps=source_fps,
            language=language,
            length_correction=False
        )


class RPS_Doc_Wikipedia_Importance_Length_Correction(  # noqa
    Base_Importance
):
    r""" Given a bag of {1,2}-wordgram model trained on Wikipedia articles p,
    and a model trained on the source domain q. This is the logarithm of the
    ratio p(doc)/q(doc). If length_correction is enabled, then the length of
    score is adjusted by adding the term log(p_poisson(len) / q_poisson(len))
    to the final score. Corrects for length by adding a length penalty term.
    """
    __slots__ = ()

    def __init__(
            self,
            wiki_fps: Tuple[str, str],
            source_fps: Tuple[str, str],
            language: str,
            *args, **kwargs  # noqa
    ):
        super(RPS_Doc_Wikipedia_Importance_Length_Correction,
              self).__init__(
            target_fps=wiki_fps,
            source_fps=source_fps,
            language=language,
            length_correction=True
        )


class RPS_Doc_Books_Importance(Base_Importance):  # noqa
    r""" Given a bag of {1,2}-wordgram model trained on Books p,
    and a model trained on the source domain q. This is the logarithm of the
    ratio p(doc)/q(doc). If length_correction is enabled, then the length of
    score is adjusted by adding the term log(p_poisson(len) / q_poisson(len))
    to the final score.
    """
    __slots__ = ()

    def __init__(
            self,
            books_fps: Tuple[str, str],
            source_fps: Tuple[str, str],
            language: str,
            *args, **kwargs  # noqa
    ):
        super(RPS_Doc_Books_Importance, self).__init__(
            target_fps=books_fps,
            source_fps=source_fps,
            language=language,
            length_correction=False
        )


class RPS_Doc_Books_Importance_Length_Correction(  # noqa
    Base_Importance
):  # noqa
    r""" Given a bag of {1,2}-wordgram model trained on Books p,
    and a model trained on the source domain q. This is the logarithm of the
    ratio p(doc)/q(doc). If length_correction is enabled, then the length of
    score is adjusted by adding the term log(p_poisson(len) / q_poisson(len))
    to the final score. Corrects for length by adding a length penalty term.
    """
    __slots__ = ()

    def __init__(
            self,
            books_fps: Tuple[str, str],
            source_fps: Tuple[str, str],
            language: str,
            *args, **kwargs  # noqa
    ):
        super(RPS_Doc_Books_Importance_Length_Correction, self).__init__(
            target_fps=books_fps,
            source_fps=source_fps,
            language=language,
            length_correction=True
        )


class RPS_Doc_OpenWebText_Importance(Base_Importance):  # noqa
    r""" Given a bag of {1,2}-wordgram model trained on OpenWebText p,
    and a model trained on the source domain q. This is the logarithm of the
    ratio p(doc)/q(doc). If length_correction is enabled, then the length of
    score is adjusted by adding the term log(p_poisson(len) / q_poisson(len))
    to the final score.
    """
    __slots__ = ()

    def __init__(
            self,
            openwebtext_fps: Tuple[str, str],
            source_fps: Tuple[str, str],
            language: str,
            *args, **kwargs  # noqa
    ):
        super(RPS_Doc_OpenWebText_Importance, self).__init__(
            target_fps=openwebtext_fps,
            source_fps=source_fps,
            language=language,
            length_correction=False
        )


class RPS_Doc_OpenWebText_Importance_Length_Correction(  # noqa
    Base_Importance):  # noqa
    r""" Given a bag of {1,2}-wordgram model trained on OpenWebText p,
    and a model trained on the source domain q. This is the logarithm of the
    ratio p(doc)/q(doc). If length_correction is enabled, then the length of
    score is adjusted by adding the term log(p_poisson(len) / q_poisson(len))
    to the final score. Corrects for length by adding a length penalty term.
    """
    __slots__ = ()

    def __init__(
            self,
            openwebtext_fps: Tuple[str, str],
            source_fps: Tuple[str, str],
            language: str,
            *args, **kwargs  # noqa
    ):
        super(
            RPS_Doc_OpenWebText_Importance_Length_Correction, self
        ).__init__(
            target_fps=openwebtext_fps,
            source_fps=source_fps,
            language=language,
            length_correction=True
        )
