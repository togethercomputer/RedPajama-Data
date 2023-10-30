import numpy as np
from typing import Tuple


def compute_hash(ngram: str, buckets: int):
    return int(abs(hash(ngram)) % buckets)


def hash_feature(
        unigrams: Tuple[str], bigrams: Tuple[str], buckets: int
) -> np.ndarray:
    counts = np.zeros(buckets, dtype=np.int64)

    for unigram in unigrams:
        counts[compute_hash(unigram, buckets=buckets)] += 1

    for bigram in bigrams:
        counts[compute_hash(bigram, buckets=buckets)] += 1

    return counts
