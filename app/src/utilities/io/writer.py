import pathlib
import msgspec
import gzip
import pyarrow as pa
import pyarrow.parquet as pq
from urllib.parse import urlparse
import boto3

from typing import Type, Any, Dict, List, Tuple, Optional


class Writer:
    def __init__(
            self,
            uri: str,
            schema: List[Tuple[str, type]],
            s3_client: Optional[boto3.client] = None
    ):
        self._client = s3_client
        uri = urlparse(uri)

        if uri.scheme == "s3":
            raise NotImplementedError("streaming to S3 not supported yet")

        elif uri.scheme == "file":
            fp = pathlib.Path(uri.path)

            if not fp.parent.exists():
                fp.parent.mkdir(parents=True, exist_ok=True)

            if fp.suffix == ".gz":
                self._filehandle = gzip.open(fp, mode="wb")
            elif fp.suffix == ".jsonl":
                self._filehandle = open(fp, mode="wb")
            else:
                raise ValueError(f"File type of {fp} not supported.")
        else:
            raise ValueError(f"Invalid uri: {uri}; must be of the form "
                             f"s3://<bucket>/<key> or file://<path>")

        # encode records using msgspec
        self._encoder = msgspec.json.Encoder()
        self._buffer = bytearray(64)

        # define record struct
        self._record: Type[msgspec.Struct] = msgspec.defstruct(
            name="Record", fields=schema
        )

    def write(self, data_obj: Dict[str, Any], flush: bool = False):
        self._encoder.encode_into(self._record(**data_obj), self._buffer)
        self._buffer.extend(b"\n")
        self._filehandle.write(self._buffer)

        if flush:
            self.flush()

    def close(self):
        self.flush()
        self._filehandle.close()

    def flush(self):
        self._filehandle.flush()
        self._buffer.clear()


class ParquetBatchWriter:

    def __init__(self, output_fp, schema: pa.Schema):
        self._schema = schema
        self._writer = pq.ParquetWriter(output_fp, self._schema)
        self.__init_batch()

    def close(self):
        if len(self._batch[self._schema.names[0]]) > 0:
            self.write_batch()
        self._writer.close()

    def update_batch(self, obj: Dict[str, Any]):
        for col in self._schema.names:
            self._batch[col].append(obj[col])

    def write_batch(self):
        self._writer.write_batch(batch=pa.record_batch(
            data=[
                pa.array(self._batch[field.name], type=field.type)
                for field in self._schema
            ],
            schema=self._schema
        ))
        self.__init_batch()

    def __init_batch(self):
        self._batch = {col: [] for col in self._schema.names}
