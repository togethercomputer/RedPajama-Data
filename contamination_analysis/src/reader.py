import io
from datetime import datetime
import jsonlines
import pathlib
import zstandard
from typing import List, Union


def get_timestamp():
    return datetime.now().isoformat()


# modified version of the reader in lm_dataformat
# reference:
# https://github.com/leogao2/lm_dataformat/blob/master/lm_dataformat/__init__.py
class Reader:
    """ Reads jsonl and jsonl.zst files.

    TODO: extend support for gzip compressed files

    @param input_files: path to input file or list of paths to input files
    """

    def __init__(
            self, input_files: Union[pathlib.Path, List[pathlib.Path]]):
        if isinstance(input_files, pathlib.Path):
            input_files = [input_files]

        self.input_files = input_files

    def read(self, text_key: str = "text", yield_meta: bool = False):
        for file in self.input_files:
            if str(file).endswith(".jsonl"):
                yield from _read_jsonl(
                    file, text_key=text_key, yield_meta=yield_meta
                )
            elif str(file).endswith(".jsonl.zst"):
                yield from _read_jsonl_zst(
                    file, text_key=text_key, yield_meta=yield_meta
                )
            else:
                raise ValueError(f"File {file} is not"
                                 f" a jsonl or jsonl.zst file.")


def _handle_jsonl(
        jsl_reader: jsonlines.Reader,
        text_key: str = "text",
        yield_meta: bool = False,
):
    idx = 0
    for obj in jsl_reader:
        text = obj[text_key]
        idx += 1

        if yield_meta:
            meta = obj["meta"] if "meta" in obj else {}
            yield text, meta
            continue

        yield text


def _read_jsonl(
        fp: pathlib.Path,
        text_key: str = "text",
        yield_meta: bool = False
):
    with jsonlines.open(fp) as jsl_reader:
        yield from _handle_jsonl(
            jsl_reader=jsl_reader,
            text_key=text_key,
            yield_meta=yield_meta
        )


def _read_jsonl_zst(
        fp: pathlib.Path,
        text_key: str = "text",
        yield_meta: bool = False,
):
    with open(fp, mode="rb") as fh:
        cctx = zstandard.ZstdDecompressor()
        reader = io.BufferedReader(cctx.stream_reader(fh))
        jsl_reader = jsonlines.Reader(reader)
        yield from _handle_jsonl(
            jsl_reader=jsl_reader,
            text_key=text_key,
            yield_meta=yield_meta
        )
