def form_ngrams(sequence, n):
    history = []
    # build the first ngram, yielding only when we have a full ngram
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1

    # yield each ngram we have, then add the next item and repeat
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]
