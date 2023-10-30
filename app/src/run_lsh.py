import argparse
from datetime import datetime as dt
import gc
import logging
import networkit.components as nk_components
import networkit.graph as nk_graph
import numpy as np
import os
from pathlib import Path
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import s3fs
import time
from typing import Dict, Tuple, List
from urllib.parse import urlparse

from dedupe.utils import optimal_param

LOG_FMT = '[%(asctime)s]::(PID %(process)d)::%(levelname)-2s::%(message)s'


class LSH:
    r""" Locality Sensitive Hashing (LSH) algorithm for near deduplication. """
    __slots__ = (
        "_args", "_num_bands", "_job_id", "_logger", "_sig_key", "_schema"
    )

    # regex to extract filepaths from source file listings
    fp_pattern = r'\/(\d{4}-\d{2}\/\d{4}\/.*\.json\.gz)$'

    # signature key
    sig_key_pat = "signature_sim{s}"

    def __init__(self):
        self._job_id = dt.now().strftime("%Y%m%d_%H%M%S")
        self._args = self.__parse_arguments()

        self._sig_key = self.sig_key_pat.format(
            s=str(self._args.similarity)
        )

        # get number of bands and rows
        self._num_bands, _ = optimal_param(
            threshold=self._args.similarity, num_perm=self._args.num_perm
        )

        # init schema
        self._schema = self.__init_schema()

        # init logging
        self.__init_logger()

        # log setup
        self._logger.info("=" * 80)
        self._logger.info("LSH config:")
        for k, v in vars(self._args).items():
            self._logger.info(f"{k}: {v}")
        self._logger.info("=" * 80)

    def __init_schema(self) -> pa.Schema:
        return pa.schema([
            ("id", pa.string()),
            ("shard_id", pa.string()),
            ("id_int", pa.uint64()),
            (self._sig_key, pa.list_(pa.binary()))
        ])

    def __init_logger(self):
        self._logger = logging.getLogger(self._job_id)
        self._logger.setLevel(logging.DEBUG)

        # log to file
        logfile = Path(self._args.output_dir) / "logs" / f"{self._job_id}.log"
        if not logfile.parent.exists():
            logfile.parent.mkdir(parents=True)
        filehandler = logging.FileHandler(logfile)
        filehandler.setFormatter(logging.Formatter(LOG_FMT))
        self._logger.addHandler(filehandler)

        # log to stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(LOG_FMT))
        self._logger.addHandler(stream_handler)

    def __parse_arguments(self) -> argparse.Namespace:

        if self.__doc__ is not None:
            description = " - " + self.__doc__
        else:
            description = self.__class__.__name__

        parser = argparse.ArgumentParser(
            prog=self.__class__.__name__, description=description
        )
        parser.add_argument(
            "--listings", type=str, default=None,
            help="file containing paths to minhash parquet files. LSH will be"
                 "run on the minhashes stored in these files."
        )
        parser.add_argument(
            "--input_base_uri", type=str, default=None,
            help="base uri of the input files."
        )
        parser.add_argument(
            "--output_dir", type=str, default=None,
            help="root directory where the output will be stored."
        )
        parser.add_argument(
            "--similarity", type=float, default=None,
            help="similarity threshold for two documents to be considered near"
                 " duplicates."
        )
        parser.add_argument(
            "--num_perm", type=int, default=None,
            help="number of permutations used during minhashing."
        )
        parser.add_argument(
            "--max_docs", type=int, default=-1,
            help="maximum number of documents to process. If set to -1, all "
                 "documents will be processed."
        )

        # s3
        parser.add_argument(
            "--s3_profile", type=str, default="default",
            help="aws profile to use when connecting to s3."
        )
        parser.add_argument(
            "--endpoint_url", type=str, default=None,
            help="endpoint url of the s3 server."
        )

        args = parser.parse_args()

        return args

    def __build_dataset(self) -> pa.dataset.Dataset:
        base_uri = urlparse(self._args.input_base_uri)

        if base_uri.scheme == "file":
            return self.__buil_dataset_local(base_uri)
        elif base_uri.scheme == "s3":
            return self.__build_dataset_s3()
        else:
            raise ValueError(f"Invalid base uri: {base_uri}")

    def __buil_dataset_local(self, base_uri) -> pa.dataset.Dataset:
        root_path = Path(base_uri.path)

        # 1) get paths and build pyarrow dataset
        with open(self._args.listings, "r") as f:
            input_paths = [
                root_path / Path(line.strip()) for line in f.readlines()
            ]

        return ds.dataset(
            source=input_paths, schema=self._schema, format="parquet"
        )

    def __build_dataset_s3(self) -> pa.dataset.Dataset:
        fs = s3fs.S3FileSystem(
            profile=self._args.s3_profile,
            endpoint_url=self._args.endpoint_url
        )

        # 1) get paths and build pyarrow dataset
        with open(self._args.listings, "r") as f:
            input_paths = list(map(
                lambda ln: os.path.join(self._args.input_base_uri, ln.strip()),
                f.readlines()
            ))

        return ds.dataset(
            source=input_paths, filesystem=fs, schema=self._schema,
            format="parquet"
        )

    def run(self):
        global_start_time = time.time()

        # 1) build pyarrow dataset; this is a lazy operation pointing to a
        # collection of parquet files on disk or in an S3 bucket
        pa_dset = self.__build_dataset()

        # 2) build edges
        step_time = time.time()
        self._logger.info("Start building edges")
        edges = self.__build_edges(pa_dset=pa_dset)
        step_time = time.time() - step_time
        self._logger.info(
            f"Building edges complete. Shape={edges.shape}; Time={step_time}s"
        )

        # 3) detect components
        step_time = time.time()
        self._logger.info("Start detecting components")
        (
            components, num_nodes, reversed_mapper
        ) = self.__run_connected_components(edges=edges)
        step_time = time.time() - step_time
        self._logger.info(
            f"Connected compontents complete. Time={step_time}s"
        )

        del edges
        gc.collect()

        # 4) collect cluster ids
        step_time = time.time()
        self._logger.info("Start collecting cluster ids")
        cluster_ids = self.__get_doc_to_cluster_array(
            components=components, reversed_mapper=reversed_mapper
        )
        step_time = time.time() - step_time
        self._logger.info(f"Building doc->cluster index complete. "
                          f"Time={step_time}s")

        # 5) build cluster dataframes
        step_time = time.time()
        self._logger.info("Start building final cluster dataframes")
        cluster_dataframes = self.__build_cluster_dataframes(
            pa_dset=pa_dset, doc_to_cluster=cluster_ids
        )
        step_time = time.time() - step_time
        self._logger.info(f"Building final cluster dataframes complete. "
                          f"Time={step_time}s")

        # 6) write cluster dataframes to disk
        out_root = Path(self._args.output_dir)
        for k, v in cluster_dataframes.items():

            tag = Path(k.split(".")[0]).with_suffix(".clusters.parquet")
            if not (out_root / tag).parent.exists():
                (out_root / tag).parent.mkdir(parents=True)

            # write to disk
            v.write_parquet(out_root / tag)
            self._logger.info(f"Wrote cluster data to {out_root / tag}")

        elapsed_time = time.time() - global_start_time
        self._logger.info(f"LSH complete. Total time: {elapsed_time}s")

    def __build_edges(self, pa_dset: pa.dataset.Dataset) -> np.ndarray:

        # build polars query plan
        query = pl.scan_pyarrow_dataset(pa_dset)

        if self._args.max_docs > 0:
            query = query.head(self._args.max_docs)

        query = (
            query
            .select(
                pl.col(["id_int", self._sig_key])
            )
            .filter(
                ~pl.col(self._sig_key).is_null()
            )
            .with_columns(
                pl.Series(
                    name="band",
                    values=[list(range(self._num_bands))],
                    dtype=pl.List(pl.UInt8)
                )
            )
            .explode(self._sig_key, "band")
            .group_by(self._sig_key, "band")
            .agg(pl.col("id_int"))
            .filter(
                pl.col("id_int").list.lengths() > 1
            )
            .select(
                pl.col("id_int"),
                pl.col("id_int").list.min().alias("min_node")
            )
            .explode("id_int")
            .filter(
                pl.col("id_int") != pl.col("min_node")
            )
            .select(
                pl.concat_list(["id_int", "min_node"]).alias("edges")
            )
            .unique("edges")
        )

        self._logger.debug(f"Query Plan:\n{query.explain()}")
        self._logger.debug(f"Start running query...")
        edges = query.collect(streaming=True).to_numpy().flatten()
        self._logger.debug(f"Completed running query.")
        gc.collect()

        return edges

    @staticmethod
    def __run_connected_components(
            edges: np.ndarray
    ) -> Tuple[List[List[int]], int, Dict[int, int]]:
        # build graph from edges
        graph = nk_graph.Graph()
        node_mapper = {}

        for row in edges:
            node_id1, node_id2 = row

            if node_id1 not in node_mapper:
                node_mapper[node_id1] = graph.addNode()

            if node_id2 not in node_mapper:
                node_mapper[node_id2] = graph.addNode()

            graph.addEdge(node_mapper[node_id1], node_mapper[node_id2])

        reversed_mapper = {value: key for key, value in node_mapper.items()}

        # compute connected components
        cc = nk_components.ConnectedComponents(G=graph)
        cc.run()
        components = cc.getComponents()
        num_nodes = sum(cc.getComponentSizes().values())

        return components, num_nodes, reversed_mapper

    @staticmethod
    def __get_doc_to_cluster_array(
            components: List[List[int]], reversed_mapper: Dict[int, int]
    ) -> np.ndarray:
        def __process_comp(comp) -> np.ndarray:
            nodes = np.array(
                list(map(reversed_mapper.get, comp))
            ).reshape(-1, 1)
            cluster_id = min(map(reversed_mapper.get, comp))
            cluster_id = np.repeat(cluster_id, len(nodes)).reshape(-1, 1)
            return np.hstack((nodes, cluster_id))

        data = np.vstack(tuple(map(__process_comp, components)))

        return data

    def __build_cluster_dataframes(
            self, pa_dset: pa.dataset.Dataset, doc_to_cluster: np.ndarray
    ) -> Dict[str, pl.DataFrame]:
        cluster_df = pl.LazyFrame(
            data=doc_to_cluster,
            schema=[("id_int", pl.UInt64), ("cluster_id", pl.UInt64)]
        )

        # build polars query plan
        query = pl.scan_pyarrow_dataset(pa_dset)

        if self._args.max_docs > 0:
            query = query.head(self._args.max_docs)

        partitioned_dfs = (
            query
            .select(pl.col(["id", "id_int", "shard_id"]))
            .join(other=cluster_df, on="id_int", how="inner")
            .select(pl.col(["id", "id_int", "cluster_id", "shard_id"]))
            .collect()
        )

        with pl.Config(set_fmt_str_lengths=5000, tbl_rows=20):
            self._logger.info(
                f"First 20 rows of minhash clusters:\n\n"
                f"{partitioned_dfs.sort(by='cluster_id').head(20)}"
            )
        time.sleep(2)

        partitioned_dfs = partitioned_dfs.partition_by(by="shard_id",
                                                       as_dict=True)

        return partitioned_dfs


if __name__ == '__main__':
    job = LSH()
    job.run()
