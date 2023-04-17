from datetime import datetime
import fasttext
import re
from typing import List, Tuple


def get_timestamp() -> str:
    return datetime.now().isoformat()


def predict_lang(
        text: str, lang_model: fasttext.FastText._FastText, k=5
) -> Tuple[List[str], List[float]]:
    r""" Predict top-k languages of text.

    @param text: text to predict language of
    @param lang_model: language model
    @param k: number of predictions to return, defaults to 5

    @return: list of predicted languages and list of corresponding
        confidence scores
    """
    # preprocess text
    text = text.lower().replace("\n", " ").replace("\t", " ")

    tags, confs = lang_model.predict(text, k=k)

    # convert confs to float
    confs = [float(conf) for conf in confs]

    # convert lang codes to names
    tags = [tag.replace("__label__", "") for tag in tags]

    return tags, confs


def format_arxiv_id(arxiv_id: str) -> str:
    r""" this function brings the raw arxiv-id into a format compliant with the
    specification from arxiv. This is used to create the url to the arxiv
    abstract page.

    - Format prior to March 2007:
        <archive>/YYMMNNN where N is a 3-digit number
    - Format after March 2007: <archive>/YYMM.NNNNN where N is a 5 (or 6)-digit
        number

    References: https://info.arxiv.org/help/arxiv_identifier.html

    @param arxiv_id: raw arxiv id which can be in one of the following formats:
        - <archive><YY><MM><NNN>
        - <YY><MM><NNNNN|NNNNNN>

    @return: formatted arxiv id
    """
    match = re.search(r'^([a-zA-Z-]*)([\d\.]+)$', arxiv_id)

    if match is None:
        raise ValueError(f"Invalid arxiv id: {arxiv_id}")

    if match.group(1) == "":
        return match.group(2)

    return f"{match.group(1)}/{match.group(2)}"
