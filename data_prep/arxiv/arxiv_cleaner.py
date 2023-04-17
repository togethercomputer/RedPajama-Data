import concurrent.futures
from datetime import datetime
import fasttext
import json
import pathlib
import tarfile
from typing import List, Tuple, Dict, Union
import gzip
import tempfile
import uuid
import re

from utils import predict_lang, get_timestamp, format_arxiv_id

# suppress fasttext warning
fasttext.FastText.eprint = lambda x: None

# constants
ARXIV_URL = "https://arxiv.org/abs/"
FT_MODEL_PATH = "models/lid.176.bin"


class ArxivCleaner:
    r""" Class for cleaning raw arxiv data. """

    def __init__(
            self,
            data_dir: pathlib.Path,
            work_dir: pathlib.Path,
            target_dir: pathlib.Path,
            worker_id: str = None
    ):
        self._data_dir = data_dir
        self._work_dir = work_dir
        self._target_dir = target_dir
        self._worker_id = worker_id if worker_id else str(uuid.uuid4())

        # make sure dirs exist
        for d in [self._work_dir, self._target_dir]:
            if not d.exists():
                d.mkdir(parents=True)

    def run_parallel(
            self, max_files: int = None, workers: int = None,
            tar_fp_list: List[str] = None
    ):
        r""" function to run the cleaning process in parallel. This function
        will iterate over all arxiv projects and clean the tex files. The
        cleaned tex files are then written to a jsonl file.
        @param max_files: maximum number of files to process, defaults to -1
            which means all files are processed. This is useful for testing.
        @param workers: number of workers to use, defaults to None which means
            that all cores are used.
        @param tar_fp_list: list of tars to process. Defaults to None which
            means that all files in data_dir are processed.
        """
        out_file = self._target_dir / f"arxiv_{self._worker_id}.jsonl"
        with open(out_file, "w") as f:
            with concurrent.futures.ProcessPoolExecutor(workers) as executor:
                for record, arxiv_id in executor.map(
                        create_record_single_arg,
                        self.arxiv_iterator(
                            max_files=max_files, tar_fp_list=tar_fp_list
                        )
                ):
                    if record is None:
                        print(f"[{get_timestamp()}][ERROR] "
                              f"failed  to process {arxiv_id}")
                        continue

                    if len(record["text"]) == 0:
                        print(f"[{get_timestamp()}][WARNING] "
                              f"empty text for {arxiv_id}")
                        continue

                    f.write(json.dumps(record) + "\n")
                    print(f"[{get_timestamp()}][INFO] "
                          f"processed {arxiv_id}")

                executor.shutdown(wait=True)

    def run(self, max_files: int = -1, out_fname: str = "arxiv.jsonl"):
        r""" function to run the cleaning process. This function will iterate
        over all arxiv projects and clean the tex files. The cleaned tex files
        are then written to a jsonl file.

        @param max_files: maximum number of files to process, defaults to -1
            which means all files are processed. This is useful for testing.
        @param out_fname: name of the output file, defaults to "arxiv.jsonl"
        """
        with open(self._target_dir / out_fname, "w") as f:
            for tex_files, yymm, arxiv_id, timestamp in self.arxiv_iterator(
                    max_files=max_files
            ):
                record, arxiv_id = create_record(
                    tex_files=tex_files,
                    yymm=yymm,
                    arxiv_id=arxiv_id,
                    timestamp=timestamp
                )

                if record is None:
                    print(f"[{get_timestamp()}][ERROR] "
                          f"failed  to process {arxiv_id}")
                    continue

                if len(record["text"]) == 0:
                    print(f"[{get_timestamp()}][WARNING] "
                          f"empty text for {arxiv_id}")
                    continue

                f.write(json.dumps(record) + "\n")
                print(f"[{get_timestamp()}][INFO] "
                      f"processed {arxiv_id}")

    def arxiv_iterator(
            self, max_files: int = -1, tar_fp_list: List[str] = None
    ):
        r""" iterator over arxiv shards. Each shard contains tex projects or
        files that are compressed using gzip. This function will extract the
        tex files and yield them together with yymm, the raw arxiv id and the
        timestamp of the project.

        @param max_files: maximum number of files to process, defaults to -1
            which means all files are processed.
        @param tar_fp_list: optional list of tar files to process, defaults to
            None. In this case all tar files in data_dir are processed.

        @return: iterator over tex files, yymm, arxiv id and timestamp.
        """
        if tar_fp_list is None:
            def _tar_fp_iterator():
                for _tar_fp in self._data_dir.glob("*.tar"):
                    yield _tar_fp
        else:
            def _tar_fp_iterator():
                for _tar_fp in tar_fp_list:
                    yield _tar_fp

        failed = 0
        processed = 0

        for tar_fp in _tar_fp_iterator():
            print(f"[{get_timestamp()}][INFO] start processing {tar_fp}")

            with tempfile.TemporaryDirectory(dir=self._work_dir) as tmpdir:
                with tarfile.open(tar_fp) as tf:
                    tf.extractall(members=tf.getmembers(), path=tmpdir)

                    for proj_dir_or_file in pathlib.Path(tmpdir).rglob("*.gz"):

                        # get arxiv id and month from the filename
                        yymm = proj_dir_or_file.parent.stem
                        arxiv_id = proj_dir_or_file.stem

                        # load the tex source files (we also get the timestamp
                        # here)
                        data = _tex_proj_loader(proj_dir_or_file)

                        if data is None:
                            failed += 1
                            continue

                        tex_files, timestamp = data
                        processed += 1

                        if processed > max_files > 0:
                            break

                        yield tex_files, yymm, arxiv_id, timestamp

                    else:
                        continue
                    break

        print(f"[{get_timestamp()}][INFO] # Failed loading : {failed}")
        print(f"[{get_timestamp()}][INFO] done.")


def create_record_single_arg(args):
    r""" convenience function to create a record from a single argument. """
    return create_record(*args)


def create_record(
        tex_files: List[str],
        yymm: str,
        arxiv_id: str,
        timestamp: float
) -> Tuple[Union[Dict[str, Union[str, Dict[str, str]]], str, None], str]:
    r""" function to create a record from the tex files, yymm, arxiv id and
    timestamp. The function also detects the language of the tex files using a
    fasttext model.

    @param tex_files: list of tex file contents as strings
    @param yymm: yymm of the arxiv project
    @param arxiv_id: raw arxiv id
    @param timestamp: timestamp of the arxiv project

    @return: dictionary containing the cleaned tex text and metadata
    """
    # clean tex files
    try:
        cleaned_str = clean_tex_files(tex_files)
    except Exception as e:
        return None, arxiv_id

    if len(cleaned_str) == 0:
        return {"text": "", "meta": {}}, arxiv_id

    # get the arxiv id in the correct format
    try:
        clean_arxiv_id = format_arxiv_id(arxiv_id)
    except Exception as e:
        print(f"[WARNING] failed to format arxiv id {arxiv_id}; excpetion={e}")
        clean_arxiv_id = arxiv_id

    # detect language
    ft_model = fasttext.load_model(path=str(FT_MODEL_PATH))
    lang, _ = predict_lang(text=cleaned_str, lang_model=ft_model, k=1)

    try:
        lang = lang[0]
    except IndexError:
        lang = "unknown"

    if timestamp is not None:
        timestamp = datetime.fromtimestamp(timestamp).isoformat()

    return (
        {
            "text": cleaned_str,
            "meta": {
                "timestamp": timestamp,
                "yymm": yymm,
                "arxiv_id": clean_arxiv_id,
                "language": lang,
                "url": f"{ARXIV_URL}{clean_arxiv_id}",
                "source": "arxiv"
            }
        },
        clean_arxiv_id
    )


def _tex_proj_loader(
        file_or_dir_path: pathlib.Path
) -> Union[Tuple[List[str], float], None]:
    r""" function to load the tex files from a tar file or a gzip file. The
    function will return a tuple containing a list of tex files and the
    timestamp of the project.

    @param file_or_dir_path: path to the tar file or the gzip file

    @return: tuple containing a list of tex files and the timestamp of the
        project
    """
    files_and_content = []

    timestamp = file_or_dir_path.lstat().st_mtime

    try:
        # if it is a directory, open it as a tarfile
        with tarfile.open(file_or_dir_path) as sub_tf:
            for member in sub_tf.getmembers():
                if member.name.endswith(".tex"):

                    file_content = sub_tf.extractfile(member).read()

                    try:
                        file_content = file_content.decode("utf-8")
                    except UnicodeDecodeError:
                        print(f"[{get_timestamp()}][ERROR] "
                              f"UnicodeDecodeError: {file_or_dir_path}")
                        return None

                    files_and_content.append(file_content)

    except tarfile.ReadError:
        # otherwise we try opening it as a gzip file
        try:
            with gzip.open(file_or_dir_path, "rb") as gz:
                file_content = gz.read()
        except Exception as e:
            # all fails, we skip this file
            print(f"[ERROR] {e}: {file_or_dir_path}")
            return None

        try:
            file_content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            print(f"[{get_timestamp()}][ERROR] "
                  f"UnicodeDecodeError: {file_or_dir_path}")
            return None

        files_and_content.append(file_content)

    except Exception as e:
        print(f"[ERROR] {e}: {file_or_dir_path}")
        return None

    return files_and_content, timestamp


def clean_tex_files(tex_files: List[str]) -> str:
    r""" function takes a list of tex files and returns a cleaned version of
    the tex project. The cleaned version is a concatenation of the tex files
    with the following modifications:

    - if multiple latex files, then concatenate them
    - remove all comments (i.e. all lines starting with %)
    - remove everything before the first \section header
    - remove everything after the first occurrence of either \appendix or
        \bibliography
    - inline-expand definitions and macros

    @param tex_files: list of file_content strings

    @return: cleaned tex project as a string, empty string if no tex files are
        provided
    """
    if len(tex_files) == 0:
        return ""

    # build dictionaries that contain the definitions of all macros in all tex
    # files. This is later used to expand all macros used in the text with
    # their definitions, so that consistency among different authors is
    # ensured.

    non_arg_macros = {}
    for file_content in tex_files:
        non_arg_macros.update(_build_non_arg_macros_dict(file_content))

    # TODO: macros that take arguments are not supported yet
    arg_macros = {}

    # join multiple latex files with a newline character
    cleaned_latex_file_str = "\n".join(
        _clean_tex_file(
            file_content=file_content,
            arg_macros=arg_macros,
            non_arg_macros=non_arg_macros
        )
        for file_content in tex_files
    )

    return cleaned_latex_file_str


def _clean_tex_file(
        file_content: str, arg_macros: Dict, non_arg_macros: Dict
) -> str:
    r""" function takes a tex file as input and returns a cleaned version. The
     cleaned version is a concatenation of the tex files with the
    following modifications:

    - remove all comments (i.e. all lines starting with %)
    - remove everything before the first section-like header
    - remove everything after the first occurrence of either \appendix or
        \bibliography
    - inline-expand definitions and macros

    @param file_content: the content of the tex file as a string.

    @return: cleaned tex file as a string
    """
    # find the first occurence of a \section-like header and replace everything
    # before it with an empty string. This matches the following pattern:
    #   \<section-type>[optional-args]{name}
    pattern = r"^(.*?)("
    pattern += r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
    pattern += r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
    pattern += r")"

    # if no section like header is found, then we return an empty string
    if not re.search(pattern, file_content, flags=re.DOTALL):
        return ""

    # replace everything with the second group of the match (i.e. everything
    # after and including the section header)
    file_content = re.sub(
        pattern=pattern,
        repl=r"\2",
        string=file_content,
        flags=re.DOTALL  # make sure that the dot matches also newlines
    )

    # remove all line comments
    file_content = re.sub(
        pattern=r"(?m)^%.*\n?",
        repl=r"",
        string=file_content,
        flags=re.MULTILINE
    )

    # remove all in comments within a line
    file_content = re.sub(
        # pattern matches a "%" that is not preceded by a backslash (=comment)
        pattern=r"[^\\]%.+$",
        repl=r"",
        string=file_content,
        flags=re.MULTILINE
    )

    # find the first occurence of either \appendix or \bibliography and
    # replace everything after it with an empty string
    pattern = r"("
    pattern += r"\\appendix|"
    pattern += r"\\begin\{references\}|"
    pattern += r"\\begin\{REFERENCES\}|"
    pattern += r"\\begin\{thebibliography\}|"
    pattern += r"\\bibliography\{.*\}"
    pattern += r").*$"

    file_content = re.sub(
        pattern=pattern,
        repl=r'',
        string=file_content,
        flags=re.DOTALL  # make sure that the dot matches also newlines
    )

    # inline-expand all non-arg macros
    for macro_name, macro_value in non_arg_macros.items():
        file_content = re.sub(
            # make pattern grouped to make sure that the macro is not part
            # of a longer alphanumeric word
            pattern=r"(" + macro_name + r")" + r"([^a-zA-Z0-9])",
            # replace the macro with its value and add back the character that
            # was matched after the macro
            repl=macro_value + r"\2",
            string=file_content
        )

    # inline-expand all macros that use args
    # TODO: inline-expand macros with args
    for macro_name, macro_value in arg_macros.items():
        pass

    return file_content


def _build_non_arg_macros_dict(file_content: str) -> Dict[str, str]:
    r""" function takes the content of a tex file and returns a dictionary
    that contains the definitions of all macros that do not use arguments.
    The dictionary is of the form {macro_name: macro_value}.

    @param file_content: the content of the tex file as a string.

    @return: dict
    """
    # regex for extracting \newcommand macros without arguments
    non_arg_nc_reg = re.compile(
        # this regex matches the following:
        # \newcommand{\macro_name}{macro_value}
        # \newcommand*{\macro_name}{macro_value}
        # where macro_name is only allowed to contain letters and numbers;
        # macro_value can contain any character.
        pattern=r'\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$',
        flags=re.MULTILINE
    )

    # regex for extracting \def macros without arguments
    non_arg_def_reg = re.compile(
        # this regex matches the following:
        # \def\macro_name{macro_value}
        # where macro_name is only allowed to contain letters and numbers;
        # macro_value can contain any character.
        pattern=r'\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$',
        flags=re.MULTILINE
    )

    # Extract all user-defined LaTeX macros from the preamble
    macros = {}
    for reg in [non_arg_nc_reg, non_arg_def_reg]:
        for match in reg.finditer(file_content):
            # convert the macro name and value to a raw string that can be
            # used in re.sub
            macro_name = match \
                .group(1).encode("unicode-escape").decode("utf-8")
            macro_val = match \
                .group(2).encode("unicode-escape").decode("utf-8")

            macros[macro_name] = macro_val

    return macros
