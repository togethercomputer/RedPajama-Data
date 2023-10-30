from core.document import Document


def preprocess_quality_classifier(document: Document):
    r""" Preprocesses a document for quality classification. This function
    removes all newlines and trailing whitespaces from the document.

    Args:
        document: A document.

    Returns:
        A string.
    """
    # remove newlines and trailing and leading whitespaces
    return " ".join(document.raw_content.splitlines()).strip()
