"""
Embed each row of a `.jsonl` file using a HuggingFace model and save the embeddings.

Authors: The Meerkat Team (Karan Goel, Sabri Eyuboglu, Arjun Desai)
License: Apache License 2.0
"""
import os
from argparse import ArgumentParser

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json
import torch
import torch.nn.functional as F
from rich import print
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import meerkat as mk


class TruncatedDataset:
    def __init__(
        self,
        df: mk.DataFrame,
        tokenizer: AutoTokenizer,
        chunk_size: int,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df[idx]
        tokenized = self.tokenizer(
            data["text"],
            pad_to_multiple_of=self.chunk_size,
            padding=True,
        )
        return {
            "input_ids": torch.tensor(tokenized["input_ids"][: self.chunk_size]),
            "attention_mask": torch.tensor(
                tokenized["attention_mask"][: self.chunk_size]
            ),
            "doc_id": data["id"],
            "chunk_id": 0,
        }


def create_model_and_tokenizer(
    model_name: str,
    cache_dir: str,
):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Loading model...")
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).cuda()

    return model, tokenizer


def prepare(feature_dir: str, savepath: str):
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    if os.path.exists(savepath):
        exit()


def load_dataframe(path):
    print("Loading dataframe...")
    # Load in the JSON.
    df = mk.from_json(
        path,
        lines=True,
        backend="arrow",
        read_options=pa.json.ReadOptions(**{"block_size": 10 << 20}),
    )

    if "meta" in df.columns:
        struct_array = df["meta"].data
        result = {}
        for field_index in range(struct_array.type.num_fields):
            field = struct_array.type.field(field_index)
            result[field.name] = mk.ArrowScalarColumn(
                pc.struct_field(struct_array, field.name)
            )
        meta_df = mk.DataFrame(result)
    else:
        meta_df = mk.DataFrame()

    if "id" in meta_df.columns:
        df["id"] = meta_df["id"]
    elif "arxiv_id" in meta_df.columns:
        df["id"] = meta_df["arxiv_id"]
    else:
        try:
            df["id"] = meta_df["pkey"]
        except:
            df.create_primary_key("id")
    df = df.set_primary_key("id")

    try:
        df = df.drop("pkey")
    except ValueError:
        pass

    assert set(df.columns) >= set(
        ["id", "text"]
    ), f"Unexpected columns: {set(df.columns)}"
    return df


def create_dataloader(
    filepath: str,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    batch_size: int,
    num_workers: int,
):
    dataset = TruncatedDataset(
        load_dataframe(filepath),
        tokenizer,
        chunk_size=chunk_size,
    )
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Extract features from the model."""
    # Extract features from the model
    attention_mask = attention_mask.cuda()
    outputs = model.forward(input_ids.cuda(), attention_mask=attention_mask)[0]

    # Use the attention mask to average the output vectors.
    outputs = outputs.cpu()
    attention_mask = attention_mask.cpu()
    features = (outputs * attention_mask.unsqueeze(2)).sum(1) / attention_mask.sum(
        1
    ).unsqueeze(1).cpu()

    # Normalize embeddings
    features = F.normalize(features, p=2, dim=1).numpy()

    return features


def run_feature_extraction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    print("Feature extraction...")
    storage = []
    for batch in tqdm(dataloader):
        features = extract_features(model, batch["input_ids"], batch["attention_mask"])
        storage.append(features)

    # Save the features to disk.
    return np.concatenate(storage, axis=0).reshape(-1, 384)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--cache_dir", type=str, default="/home/karan/models/")
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=f"/home/karan/data/pyjama/features/",
    )

    args = parser.parse_args()
    feature_dir = os.path.join(args.feature_dir, args.model_name)

    CUDA_VISIBLE_DEVICES = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)

    # Get num_gpus on this machine.
    num_gpus = torch.cuda.device_count()

    filepath = args.filepath
    filename = os.path.basename(filepath)
    savepath = os.path.join(feature_dir, filename.replace(".jsonl", ".npy"))
    prepare(feature_dir, savepath)

    model, tokenizer = create_model_and_tokenizer(args.model_name, args.cache_dir)
    dataloader = create_dataloader(
        filepath,
        tokenizer,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    features = run_feature_extraction(model, dataloader)
    np.save(savepath, features)
    print("Done.")
