import os
import pandas as pd
from tqdm import tqdm

BASE_URL="https://archive.org/download/stackexchange/"

table = pd.read_html(BASE_URL)[0]
sources = [x.replace(" (View Contents)", "") for x in table['Name'].tolist()]
sources = [x for x in sources if x.endswith(".7z")]
for source in tqdm(sources):
    # if ".meta." not in source:
    print(f"source: {source}")
    os.system("wget "+BASE_URL+source+" -O "+"./data/"+source)
    os.system("7z x ./data/"+source+" -o./data/"+source[:-3])
    os.system(f"mv ./data/{source[:-3]}/Posts.xml ./data/{source[:-3]}.xml")
    os.system(f"rm -rf ./data/{source[:-3]}")
    os.system(f"rm ./data/{source}")
