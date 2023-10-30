"""
To reduce dependencies, the functions in this module are adapted from
the `datasketch` library with minor modifications.
"""

from scipy.integrate import quad as integrate
import hashlib
import numpy as np
import struct
from typing import Iterable, List

from utilities.text import form_ngrams


def _false_positive_probability(threshold, b, r):
    def proba(s):
        return 1 - (1 - s ** float(r)) ** float(b)

    a, *_ = integrate(proba, 0.0, threshold)

    return a


def _false_negative_probability(threshold, b, r):
    def proba(s):
        return 1 - (1 - (1 - s ** float(r)) ** float(b))

    a, *_ = integrate(proba, threshold, 1.0)

    return a


def optimal_param(
        threshold: float,
        num_perm: int,
        false_positive_weight: float = 0.5,
        false_negative_weight: float = 0.5
):
    r"""
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def sha1_hash32(data: bytes) -> int:
    """
    A 32-bit hash function based on SHA1.

    Note:
        This implementation is copied from datasketch to avoid dependency.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def generate_signature(
        words_sequence: Iterable[str],
        ngram_size: int,
        permutations: np.ndarray,
        max_hash: int,
        mersenne_prime: int
) -> np.ndarray:
    r"""
    Combined with some datasketch code to better parallelize computation.

    Note:
        This implementation is adapted from the near-dedupe implementation by
        the bigcode project.

    Parameters
    ----------
    words_sequence : str
        A sequence of (normalized) words for which to generate a signature.
    ngram_size : int
        The size of n-grams.
    permutations : np.ndarray
        The permutations for the minhash.
    max_hash: int
        The maximum value for hashes.
    mersenne_prime: int
        The mersenne prime.

    Returns
    -------
    List[np.uint32]
        The minhash signature.
    """
    num_perm = permutations.shape[-1]
    hashvalues = np.ones(num_perm, dtype=np.uint64) * max_hash
    tokens = {" ".join(t) for t in form_ngrams(words_sequence, ngram_size)}
    h_vals = np.array(
        [sha1_hash32(token.encode("utf-8")) for token in tokens],
        dtype=np.uint64
    )
    a, b = permutations
    phv = np.bitwise_and(
        ((h_vals * np.tile(a, (len(h_vals), 1)).T).T + b) % mersenne_prime,
        max_hash
    )

    # compute the minhash
    signature = np.vstack([phv, hashvalues]).min(axis=0).astype(np.uint32)

    return signature
