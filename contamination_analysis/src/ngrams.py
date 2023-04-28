"""

Utility function used to compute ngrams in a sequence.

Implementation taken from nltk source / lm-evaluation-harness
- https://www.nltk.org/_modules/nltk/util.html
- https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/decontamination/janitor.py

"""


def form_ngrams(sequence, n):
    history = []
    while n > 1:
        # PEP 479, prevent RuntimeError from being raised when StopIteration
        # bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]
