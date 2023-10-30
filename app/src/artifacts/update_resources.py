import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import time
import urllib.request
import requests
import tarfile
from typing import Dict, List, Tuple

_UT1_BLACKLIST_URL = "http://dsi.ut-capitole.fr" \
                     "/blacklists/download/blacklists.tar.gz"
_LDNOOBW_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-" \
               "Naughty-Obscene-and-Otherwise-Bad-Words/master/{lang}"


def _build_category_index(raw_categories) -> Dict[Tuple[str], int]:
    r""" Build a mapping with a list of categories, corresponding to a unique
    combination of categories, to a category ID.
    """
    categories = sorted(raw_categories)

    category_index = {}
    for i, category in enumerate(itertools.chain.from_iterable(
            itertools.combinations(categories, r) for r in
            range(1, len(categories) + 1)
    )):
        category_index[tuple(str(s) for s in sorted(category))] = i

    return category_index


def _domain_to_category_list_mapping(bad_urls_dir, raw_categories):
    domain_to_category = defaultdict(list)

    for category_dir in (bad_urls_dir / "blacklists").iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        if category not in raw_categories:
            continue

        with open(category_dir / "domains", "r") as f:
            for dom in map(str.strip, f.readlines()):
                domain_to_category[dom].append(category)

    # postprocess
    domain_to_category = {
        dom: tuple(str(s) for s in sorted(set(categories)))
        for dom, categories in domain_to_category.items()
    }

    return domain_to_category


def create_bad_urls_index(artifacts_dir: Path, raw_categories: List[str]):
    r""" update the URL blacklists from the University of Toulouse:

    @param artifacts_dir: (Path) The path to the resources directory
    @param raw_categories: (List[str]) The domain categories

    """

    ut1_blacklist_dir = artifacts_dir / "bad_urls"

    if not ut1_blacklist_dir.exists():
        ut1_blacklist_dir.mkdir(parents=True)

    print(f"fetching UT1 blacklist from {_UT1_BLACKLIST_URL}...")

    with urllib.request.urlopen(_UT1_BLACKLIST_URL) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            tar.extractall(path=ut1_blacklist_dir)

    with open(ut1_blacklist_dir / "_FETCH_TIMESTAMP", "w") as f:
        f.write(str(int(time.time())))

    print(f"raw UT1 list fetched.")

    category_index = _build_category_index(raw_categories)

    # convert the raw UT1 blacklist to a domain -> category_id mapping where
    # a category corresponds to any combination of raw categories.
    domain_to_category_list = _domain_to_category_list_mapping(
        ut1_blacklist_dir, raw_categories
    )

    domain_to_category_id = {
        dom: category_index[categories]
        for dom, categories in domain_to_category_list.items()
    }

    with open(ut1_blacklist_dir / "domain_to_category_id.json", "w") as f:
        json.dump(domain_to_category_id, f)

    # save the category index as int -> category mapping
    category_index = {
        i: categories for categories, i in category_index.items()
    }
    with open(ut1_blacklist_dir / "category_index.json", "w") as f:
        json.dump(category_index, f)


def create_bad_words_list(artifacts_dir: Path, lang: str):
    r""" Fetch the LDNOOBW word list

    Args:
        artifacts_dir (Path): The path to the resources directory
        lang (str): The language to fetch the word list for
    """

    ldnoobw_dir = artifacts_dir / "bad_words"

    if not ldnoobw_dir.exists():
        ldnoobw_dir.mkdir(parents=True)

    word_list_fp = ldnoobw_dir / f"{lang}.txt"
    url = _LDNOOBW_URL.format(lang=lang)

    print(f"fetching bad words list from {url}...")

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"{response.status_code} -- {url}.")

    data = response.content.decode('utf-8')

    with open(ldnoobw_dir / f"_{lang}_FETCH_TIMESTAMP", "w") as f:
        f.write(str(int(time.time())))

    data = set(w for w in data.splitlines() if w is not None)

    with open(word_list_fp, 'w') as f:
        f.write('\n'.join(data))

    print(f"bad words list ({lang}) updated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, nargs="+")
    parser.add_argument("--artifacts_dir", type=str)
    parser.add_argument("--block_categories", type=str, nargs="+")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # fetch ut1 blacklist
    create_bad_urls_index(artifacts_dir=artifacts_dir,
                          raw_categories=args.block_categories)

    # fetch ldnoobw
    langs = set(args.langs)
    for lang in langs:
        try:
            create_bad_words_list(lang=lang, artifacts_dir=artifacts_dir)
        except Exception as e:
            print(f"Failed to fetch LDNOOBW {lang}: {e}")


if __name__ == '__main__':
    main()
