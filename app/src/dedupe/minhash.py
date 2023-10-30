import hashlib
import numpy as np
from typing import List, Dict, Optional, Tuple

from dedupe.utils import optimal_param, generate_signature


class MinHash:
    _sig_key_pat = "signature_sim{s}"

    def __init__(
            self,
            similarity_thresholds: List[float],
            ngram_size: int,
            num_permutations: int,
            seed: int,
    ):
        self._similarity_thresholds = similarity_thresholds
        self._rng = np.random.RandomState(seed)
        self._ngram_size = ngram_size

        self._bands_rows = {
            str(s): optimal_param(threshold=s, num_perm=num_permutations)
            for s in similarity_thresholds
        }

        self._hashranges = {
            self._sig_key_pat.format(s=s): self._get_hashrange(b, r)
            for s, (b, r) in self._bands_rows.items()
        }

        # init minhash artifacts
        self.__init_minhash(num_permutations)

    def __init_minhash(self, num_permutations):
        # minhash constants
        self._max_hash = np.uint64((1 << 32) - 1)
        self._mersenne_prime = np.uint64((1 << 61) - 1)
        self._permutations = np.array(
            [
                (
                    self._rng.randint(
                        1, self._mersenne_prime, dtype=np.uint64
                    ),
                    self._rng.randint(
                        0, self._mersenne_prime, dtype=np.uint64
                    ),
                )
                for _ in range(num_permutations)
            ],
            dtype=np.uint64,
        ).T

        # compute checksum for permutations
        self._checksum = hashlib.sha256(
            self._permutations.tobytes()
        ).hexdigest()

    @staticmethod
    def _get_hashrange(b, r):
        return [(i * r, (i + 1) * r) for i in range(b)]

    @property
    def similarity_thresholds(self):
        return self._similarity_thresholds

    @property
    def checksum(self):
        return self._checksum

    def compute_banded_signatures(
            self, tokens: Tuple[str]
    ) -> Dict[str, Optional[List[bytes]]]:
        if len(tokens) < self._ngram_size:
            return {k: None for k in self._hashranges.keys()}

        # compute signature
        minhashes: np.ndarray = generate_signature(
            words_sequence=iter(tokens),
            ngram_size=self._ngram_size,
            permutations=self._permutations,
            max_hash=self._max_hash,
            mersenne_prime=self._mersenne_prime
        )

        # partition signatures into bands
        signatures = {
            sig_key: [
                bytes(minhashes[start:end].byteswap().data)
                for start, end in hashrange
            ]
            for sig_key, hashrange in self._hashranges.items()
        }

        return signatures
