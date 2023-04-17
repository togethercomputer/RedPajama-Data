import faiss
import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def build_flat_index(
    xb: np.ndarray,
    d: int = 32,
):
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    return index


def load_index(
    path: str,
):
    """Load the index from a path."""
    index = faiss.read_index(path)
    return index


def load_pca(path: str):
    """Load the PCA from a path."""
    pca = faiss.read_VectorTransform(path)
    return pca


def create_model_and_tokenizer(
    model_name: str,
    cache_dir: str = None,
):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Loading model...")
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    return model, tokenizer


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Extract features from the model."""
    # Extract features from the model
    attention_mask = attention_mask
    outputs = model.forward(input_ids, attention_mask=attention_mask)[0]

    # Use the attention mask to average the output vectors.
    outputs = outputs.cpu()
    attention_mask = attention_mask.cpu()
    features = (outputs * attention_mask.unsqueeze(2)).sum(1) / attention_mask.sum(
        1
    ).unsqueeze(1).cpu()

    # Normalize embeddings
    features = F.normalize(features, p=2, dim=1).numpy()

    return features


def extract_features_single(
    text: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    chunk_size: int = 512,
):
    """Extract features from the model."""
    tokenized = tokenizer(
        [text],
        pad_to_multiple_of=chunk_size,
        padding=True,
    )
    return extract_features(
        model,
        torch.tensor(tokenized["input_ids"][:chunk_size]),
        torch.tensor(tokenized["attention_mask"][:chunk_size]),
    )


def run_feature_extraction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    print("Feature extraction...")
    storage = []
    carry = (None, None)
    for batch in tqdm(dataloader):
        features = extract_features(model, batch["input_ids"], batch["attention_mask"])
        chunk_id = np.array(batch["chunk_id"])
        doc_id = np.array(batch["doc_id"])
        if (chunk_id == 0).all():
            storage.append(features)
        elif (chunk_id == 0).any():
            # Close out the previous document.
            # Aggregate based on the document ID.
            agg = np.array(
                [features[doc_id == i].mean(axis=0) for i in np.unique(doc_id)]
            )

            # Number of chunks in the first document.
            num_chunks_first = (doc_id == doc_id[0]).sum()

            # Number of chunks in the last document.
            num_chunks_last = (doc_id == doc_id[-1]).sum()

            # Batch falls on a document boundary.
            if chunk_id[0] == 0:
                # Close out the previous document and update the carry.
                storage.append(carry[0])
                carry = (None, None)

            # Batch does not fall on a document boundary.
            if carry[0] is not None:
                # Reweight the first chunk.
                agg[0] = (agg[0] * num_chunks_first + carry[0] * carry[1]) / (
                    num_chunks_first + carry[1]
                )

            # Update the carry.
            carry = (agg[-1], num_chunks_last)

            # Put the features in storage.
            storage.append(agg[:-1])

        else:
            # All chunks should have the same document ID.
            assert (doc_id == doc_id[0]).all()
            # Aggregate.
            agg = np.mean(features, axis=0)
            # Reweight.
            agg = (agg * len(features) + carry[0] * carry[1]) / (
                len(features) + carry[1]
            )
            # Update the carry: make sure to keep track of the number of chunks.
            carry = (agg, len(features) + carry[1])

    # Save the features to disk.
    return np.concatenate(storage, axis=0).reshape(-1, 384)
