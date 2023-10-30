import re


def generate_paragraphs(text: str, remove_empty: bool = True):
    for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text):
        text_slice = text[match.start():match.end()]

        if remove_empty and not text_slice.strip():
            continue

        yield text_slice
