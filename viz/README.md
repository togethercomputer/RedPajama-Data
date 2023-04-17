# Data Exploration with Meerkat

We provide a set of tools for exploring the Github subset of the dataset using the [Meerkat](https://github.com/hazyresearch/meerkat) library. Instructions for reproducing the data processing pipeline are also provided.

## Getting Started
You will need to install the [Meerkat](https://github.com/hazyresearch/meerkat) library, along with some other dependencies. We recommend using a virtual environment to install the dependencies.

Note: if you have access to a GPU, you should install the `torch` library with CUDA support and install the `faiss-gpu` package. See the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for more details on torch installation.

```
pip install -r requirements.txt
```

## Downloading Processed Data
We provide a set of processed data files that can be used to explore the dataset.
These include a compressed version of the Github dataset, along with a set of embeddings for the code snippets.

## Running the Meerkat Application
To run the Meerkat application, run the following command:
```
mk run main.py --api-port 5002 --frontend-port 8002
```

**Note**: make sure you set the `--api-port` and `--frontend-port` to different values if you run into issues viewing the application.


# Reproducing the Data Processing Pipeline

## Download Data
Download the Github `.jsonl` files. This will be approximately `200 GB` of data.

## Embed Data
We provide a self-contained script for computing embeddings for the code snippets in the dataset. This script only supports the `sentence-transformers/all-MiniLM-L6-v2` model for computing embeddings. In order to compute embeddings,
the first `chunk_size` (default `512`) tokens of text are used.

To compute and save embeddings for a single `.jsonl` file, run the following command:
```
python embed_jsonl.py --gpu 0 --filepath <path to jsonl file> --num_workers 8 --batch_size 512 --chunk_size 512 --cache_dir <path to cache dir for model> --feature_dir <dir to save embeddings>
```
This will save the embeddings to a `.npy` file in the `feature_dir` directory. The name of the file will be the same as the name of the `.jsonl` file, but with the `.npy` extension.

You will need to run this script for each `.jsonl` file in the dataset. We recommend using a job scheduler to run the scripts in parallel (or if you are on a single machine, just run them sequentially using a runner script). Embedding a single `.jsonl` file takes around ~30 minutes on a single A100 GPU.

## Reduce with PCA
We provide a script for reducing the dimensionality of the embeddings using PCA.

To reduce the dimensionality of the embeddings, run the following command:
```
python reduce_pca32.py --feature_dir <dir to save embeddings>
```
This script will save the reduced embeddings to a single `pca32.npy` file in the `feature_dir` directory. It will also save a `pca32.faiss` file, which contains the PCA model. This model can be used to reduce the dimensionality of new embeddings.

## Index with FAISS
We provide a script for indexing the embeddings using FAISS.

To index the embeddings, run the following command:
```
python index_faiss.py --dir <dir to save embeddings>
```
This script will save two indices: a flat index to a `index_flat.faiss` file in the `dir` directory, as well as a more space-efficient IVF index to a `index_ivf.faiss` file in the `dir` directory.
