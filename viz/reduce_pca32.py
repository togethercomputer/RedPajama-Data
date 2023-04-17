import os
from argparse import ArgumentParser
from glob import glob

import faiss
import numpy as np
from tqdm.auto import tqdm


def build_pca(
    xb: np.ndarray,
    d_in: int = 384,
    d_out: int = 32,
):
    pca = faiss.PCAMatrix(d_in, d_out)
    pca.train(xb)
    return pca


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="~/data/pyjama/features/sentence-transformers/all-MiniLM-L6-v2/",
    )
    args = parser.parse_args()
    dir = os.path.expanduser(args.feature_dir)

    # Load in all the files.
    files = sorted(list(glob(f"{dir}/*.sampled.npy")))
    print(f"Loading {len(files)} files into memory...")
    arrs = [np.load(file) for file in tqdm(files)]

    # Concatenate all the arrays
    arr = np.concatenate(arrs, axis=0)
    print("Combined arr:", arr.shape)

    # Create the PCA
    pca = build_pca(arr)
    faiss.write_VectorTransform(pca, f"{dir}/pca32.faiss")

    # Apply to all vectors.
    arr_reduced = pca.apply(arr)

    # Save the reduced array.
    np.save(f"{dir}/pca32.npy", arr_reduced)
