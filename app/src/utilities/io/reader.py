import io
import msgspec
import boto3
import gzip
from urllib.parse import urlparse, ParseResult
import pathlib
from typing import Optional, Type, List, Tuple
import xopen

from core.data_types import InputSpec
from core.exceptions import *


class Reader:
    r""" Read plain jsonl, jsonl.zst and jsonl.gz files using msgspec """

    def __init__(
            self,
            schema: List[Tuple[str, type]] = None,
            input_spec: Type[msgspec.Struct] = InputSpec,
            s3_client: Optional[boto3.client] = None,
            threads: int = 1,
            logger=None
    ):
        self._client = s3_client
        self._threads = threads

        if logger is None:
            self._print = print
        else:
            self._print = logger.error

        # msgspec decoder
        if schema is not None:
            input_type = msgspec.defstruct(name="Record", fields=schema)
        else:
            input_type = input_spec

        self._obj_decoder = msgspec.json.Decoder(type=input_type)

        self._total_consumed = 0

    def read(self, uri: str, max_samples: Optional[int] = -1,
             return_idx: bool = True):
        n_samples = 0

        try:
            with self.__get_filehandle(uri) as fh:
                for idx, obj in enumerate(fh):
                    try:
                        record = self._obj_decoder.decode(obj)
                        if return_idx:
                            yield idx, record
                        else:
                            yield record
                    except Exception as e:
                        self._print(f"__SAMPLE_READ_ERROR__ {uri}/{idx}: "
                                    f"{e.__class__.__name__}: {e}")
                        continue

                    n_samples += 1

                    if n_samples >= max_samples > 0:
                        break
        except S3ReadError as e:
            raise e
        except LocalReadError:
            raise e
        except Exception as e:
            raise UnknownReadError(f"unknown __URI_READ_ERROR__ {uri}: "
                                   f"{e.__class__.__name__}: {e}")

    def __get_filehandle(self, uri: str):
        uri = urlparse(uri)

        if uri.scheme == "s3":
            return self.__get_s3_filehandle(uri)

        if uri.scheme == "file":
            return self.__get_local_filehandle(uri)

        raise ValueError(f"Invalid uri: {uri}; must be of the form "
                         f"s3://<bucket>/<key> or file://<path>")

    def __get_s3_filehandle(self, uri: ParseResult):
        assert self._client is not None, "S3 client not initialized"

        try:
            streaming_body = self._client.get_object(
                Bucket=uri.netloc, Key=uri.path.lstrip("/")
            )["Body"]
            buffer = io.BytesIO(streaming_body.read())
        except Exception as e:
            raise S3ReadError(
                f"__S3_URI_READ_ERROR__ failed reading {uri.path}: "
                f"caught exception {e.__class__.__name__}: {e}"
            )

        return gzip.open(buffer, mode="rb")

    def __get_local_filehandle(self, uri: ParseResult):
        fp = pathlib.Path(uri.path)

        try:
            if fp.suffix == ".gz":
                return xopen.xopen(fp, mode="rb", threads=self._threads)

            if fp.suffix == ".jsonl":
                return open(fp, mode="rb")
        except Exception as e:
            raise LocalReadError(
                f"__LOCAL_URI_READ_ERROR__ failed reading {uri.path}: "
                f"caught exception {e.__class__.__name__}: {e}"
            )

        raise ValueError(f"File type of {fp} not supported.")
