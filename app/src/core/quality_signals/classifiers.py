import sys
from typing import List, Tuple, Type
import fasttext

from core.constants import PRECISION, CCNET_LABEL
from core.quality_signals.base import RPSBase
from core.document import Document
from core.data_types import SignalType
from core.quality_signals.utils.classifiers import \
    preprocess_quality_classifier
from utilities.register.registry_utils import *

__all__ = [
    "register_classifier_callables", "classifier_schema"
]


def classifier_schema() -> List[Tuple[str, Type]]:
    r""" Returns a list of signal names and their data types """
    return signal_schema(module=sys.modules[__name__])


def register_classifier_callables(
        wikiref_model: str,
        palm_model: str,
        wikipedia_model: str
) -> List[RPSBase]:
    r""" Returns a list of signal functions (i.e., RPSBase instances) that
    are used to extract content signals from a document.

    Args:
        wikiref_model: A fasttext model trained on Wikipedia references.
        palm_model: A fasttext model trained on ccnet vs
            {books, openwebtext, wikipedia}.
        wikipedia_model: A fasttext model trained on Wikipedia articles.

    Returns:
        A list of signal function class instances.
    """
    return list(map(
        lambda cls: cls(
            wikiref_model=wikiref_model,
            palm_model=palm_model,
            wikipedia_model=wikipedia_model,
        ),
        get_callables_from_module(module=sys.modules[__name__])
    ))


class BaseMLSignal(RPSBase):
    __slots__ = "_ft_model"

    def __init__(self, ft_model_file: str):
        super(BaseMLSignal, self).__init__()
        if ft_model_file is None:
            self._ft_model = None
        else:
            self._ft_model = fasttext.load_model(str(ft_model_file))

    def __call__(self, document: Document) -> SignalType:
        if self._ft_model is None:
            return [(0, len(document), None)]

        if len(document.raw_content) == 0:
            return [(0, len(document), None)]

        text = preprocess_quality_classifier(document=document)
        pred = self._ft_model.predict(text=text)

        (pred_label, pred_prob) = pred
        pred_label = pred_label[0]
        pred_prob = pred_prob[0]

        if pred_label == CCNET_LABEL:
            high_quality_score = 1 - pred_prob
        else:
            high_quality_score = pred_prob

        score = round(float(high_quality_score), PRECISION)
        return [(0, len(document), score)]


class RPS_Doc_ML_Wikiref_Score(BaseMLSignal):  # noqa
    r""" Fasttext classifier prediction for the document being a Wikipedia
    reference. This is the same fasttext model as in the RedPajama-1T
    dataset."""
    __slots__ = ()

    def __init__(self, wikiref_model: str, *args, **kwargs):  # noqa
        super(RPS_Doc_ML_Wikiref_Score, self).__init__(
            ft_model_file=wikiref_model
        )


class RPS_Doc_ML_Palm_Score(BaseMLSignal):  # noqa
    r""" Fasttext classifier prediction for the document being a Wikipedia
    article, OpenWebText sample or a RedPajama-V1 book."""
    __slots__ = ()

    def __init__(self, palm_model: str, *args, **kwargs):  # noqa
        super(RPS_Doc_ML_Palm_Score, self).__init__(
            ft_model_file=palm_model
        )


class RPS_Doc_ML_Wikipedia_Score(BaseMLSignal):  # noqa
    r""" Fasttext classifier prediction for the document being a Wikipedia
    article."""
    __slots__ = ()

    def __init__(self, wikipedia_model: str, *args, **kwargs):  # noqa
        super(RPS_Doc_ML_Wikipedia_Score, self).__init__(
            ft_model_file=wikipedia_model
        )
