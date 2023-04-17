import os
from argparse import ArgumentParser

import faiss
import numpy as np


def build_index(
    xb: np.ndarray,
    d: int = 32,
):
    index = faiss.index_factory(d, "IVF100,PQ8")
    # Sample 1_000_000 vectors to train the index.
    xt = xb[np.random.choice(xb.shape[0], 1_000_000, replace=False)]
    index.train(xt)
    index.add(xb)
    return index


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="~/data/pyjama/features/sentence-transformers/all-MiniLM-L6-v2",
    )

    args = parser.parse_args()
    dir = os.path.expanduser(args.dir)

    # Load in the embeddings.
    arr = np.load(f"{dir}/pca32.npy")
    print(arr.shape)

    # Create the index.
    index = build_index(arr)
    faiss.write_index(index, f"{dir}/index_ivf100_pq8.faiss")
