"""
A Meerkat app for visualizing the Github subset of the RedPajama dataset.

Authors: The Meerkat Team (Karan Goel, Sabri Eyuboglu, Arjun Desai)
License: Apache License 2.0
"""
import numpy as np
import tempfile
from utils import extract_features_single, load_pca, create_model_and_tokenizer

import meerkat as mk
from meerkat.datasets.utils import download_url


with tempfile.TemporaryDirectory() as temp_dir:
    path = download_url(
        "https://huggingface.co/datasets/meerkat-ml/lemma/resolve/main/pca.faiss",
        temp_dir,
    )
    pca = load_pca(path)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model, tokenizer = create_model_and_tokenizer(model_name)

df = mk.read(
    "https://huggingface.co/datasets/meerkat-ml/lemma/resolve/main/filtered_08cdfa755e6d4d89b673d5bd1acee5f6.mk.tar.gz"
)


def get_full_text(text_sample: str, repo_name: str, ref: str, path: str):
    """
    Get the full text of a code sample from Github.
    """
    ref = ref.split("/")[-1]
    import requests

    return requests.get(
        f"https://raw.githubusercontent.com/{repo_name}/{ref}/{path}"
    ).text


df["text_sample"] = df["text_sample"].format(mk.format.CodeFormatterGroup())
df["full_text"] = df.defer(get_full_text).format(mk.format.CodeFormatterGroup().defer())
df["search"] = mk.ArrowScalarColumn(np.zeros(len(df)))
df["embeddings"] = df["embeddings"].format(mk.format.TextFormatterGroup())


@mk.endpoint
def search(df: mk.DataFrame, new_code: str = ""):
    """The endpoint for executing a search query."""
    if new_code != "":
        features = extract_features_single(new_code, model, tokenizer)
        pca_features = pca.apply(features)
        df["search"] = np.matmul(df["embeddings"].data, pca_features.T).squeeze()
        df.set(df)


editor = mk.gui.Editor(on_run=search.partial(df), title="Search")

# build controls for the scatter plot
NUM_PCA_COMPONENTS = 5
for i in range(NUM_PCA_COMPONENTS):
    df[f"pca_{i+1}"] = df["embeddings"][:, i]

options = [f"pca_{i+1}" for i in range(NUM_PCA_COMPONENTS)] + ["search"]
x_select = mk.gui.Select(
    options,
    value="pca_1",
)
x_control = mk.gui.html.div(
    [mk.gui.Text("X Axis"), x_select], classes="grid grid-cols-[auto_1fr] gap-2"
)
y_select = mk.gui.Select(
    options,
    value="pca_2",
)
y_control = mk.gui.html.div(
    [mk.gui.Text("Y Axis"), y_select], classes="grid grid-cols-[auto_1fr] gap-2"
)
color_select = mk.gui.Select(
    options,
    value="search",
)
color_control = mk.gui.html.div(
    [mk.gui.Text("Color"), color_select], classes="grid grid-cols-[auto_1fr] gap-2"
)
select = mk.gui.html.div(
    [x_control, y_control, color_control], classes="grid grid-cols-3 gap-8 px-10"
)

scatter = mk.gui.plotly.DynamicScatter(
    df=df,
    x=x_select.value,
    y=y_select.value,
    color=color_select.value,
    max_points=10_000,
)
gallery = mk.gui.Gallery(
    scatter.filtered_df, main_column="text_sample", tag_columns=["language"]
)
page = mk.gui.Page(
    component=mk.gui.html.div(
        [
            mk.gui.html.div(
                [editor, select, scatter],
                classes="h-screen grid grid-rows-[1fr_auto_3fr] gap-4",
            ),
            gallery,
        ],
        classes="grid grid-cols-2 gap-4 h-screen py-6",
    ),
    id="lemma",
)
page.launch()
