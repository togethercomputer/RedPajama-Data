from megatron.data.indexed_dataset import MMapIndexedDataset
from transformers import AutoTokenizer

import argparse

# get the first argument as a file name, and an output file
parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="the file name to read")
parser.add_argument("output_file", help="the file name to write")
args = parser.parse_args()

ds = MMapIndexedDataset(args.file_name)

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

num_tokens = [
    len(ds[i]) for i in range(len(ds))
]

# write it out to an output_file
with open(args.output_file, "w") as f:
    for i in num_tokens:
        f.write(f"{i}\n")

print(f'Total tokens: {sum(num_tokens)}')