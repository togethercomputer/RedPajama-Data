import argparse
import boto3
from botocore.exceptions import ClientError
import configparser
import itertools
import numpy as np
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--aws_config', type=str, help='aws config file')
parser.add_argument('--target_dir', type=str, default="./data/arxiv")
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--input', type=str,
                    help='input file from which to read keys. '
                         'This is only used when running on slurm.')
parser.add_argument('--local', action='store_true')
parser.add_argument('--setup', action='store_true',
                    help='if set, we partition the keys into chunks.')
parser.add_argument('--max_files', type=int, default=-1,
                    help='max files to download, useful for testing')
args = parser.parse_args()


class ArxivDownloader:
    def __init__(self, config_file: str):
        # import configs from config file
        configs = configparser.SafeConfigParser()
        configs.read(config_file)

        # Create S3 resource & set configs
        self.s3resource = boto3.resource(
            's3',  # the AWS resource we want to use
            aws_access_key_id=configs['DEFAULT']['ACCESS_KEY'],
            aws_secret_access_key=configs['DEFAULT']['SECRET_KEY'],
            region_name='us-east-1'  # same region arxiv bucket is in
        )

    def run(self, input_file: str, tgt_dir: pathlib.Path, max_files=-1):
        (tgt_dir / 'src').mkdir(exist_ok=True, parents=True)

        with open(input_file, 'r') as f:
            file_keys = f.readlines()

        files_downloaded = 0

        for key in file_keys:
            self.__download_file(tgt_dir=tgt_dir, key=key.strip())
            files_downloaded += 1

            if files_downloaded >= max_files > 0:
                break

    def __download_file(self, key, tgt_dir: pathlib.Path):
        print('\nDownloading s3://arxiv/{} t'
              'o {}...'.format(key, pathlib.Path(tgt_dir, key)))

        try:
            self.s3resource.meta.client.download_file(
                Bucket='arxiv',
                Key=key,
                Filename=pathlib.Path(tgt_dir, key),
                ExtraArgs={'RequestPayer': 'requester'})
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print('ERROR: ' + key + " does not exist in arxiv bucket")
            else:
                try:
                    code = e.response['Error']['Code']
                    msg = e.response['Error']['Message']
                    print(f"UNKNOWN ERROR: code={code}; msg={msg}")
                except Exception as e:
                    print("UNKNOWN ERROR for key ", key, e)


def partition_keys(
        partitions_dir: pathlib.Path, config_file: str, workers: int
):
    r"""Partitions the keys of the arxiv bucket into chunks for parallel
    download.

    @param partitions_dir: the directory to save the partition files to (will be
        created if it doesn't exist)
    @param config_file: the path to the config file containing the aws
        credentials
    @param workers: the number of workers to partition the keys into
    """
    partitions_dir = pathlib.Path(partitions_dir).absolute()
    partitions_dir.mkdir(parents=True, exist_ok=True)

    # Securely import configs from private config file
    configs = configparser.SafeConfigParser()
    configs.read(config_file)

    # Create S3 resource & set configs
    print('Connecting to Amazon S3...')
    s3resource = boto3.resource(
        's3',  # the AWS resource we want to use
        aws_access_key_id=configs['DEFAULT']['ACCESS_KEY'],
        aws_secret_access_key=configs['DEFAULT']['SECRET_KEY'],
        region_name='us-east-1'  # same region arxiv bucket is in
    )

    # Create a reusable Paginator
    paginator = s3resource.meta.client.get_paginator('list_objects_v2')

    # Create a PageIterator from the Paginator
    page_iterator = paginator.paginate(
        Bucket='arxiv',
        RequestPayer='requester',
        Prefix='src/'
    )

    # partition keys into chunks
    file_parts = np.array_split(list(
        itertools.chain(
            *[
                [
                    file['Key'] for file in page['Contents']
                    if file['Key'].endswith(".tar")
                ]
                for page in page_iterator
            ]
        )),
        indices_or_sections=workers
    )

    # save chunks to disk as text files
    for i, part in enumerate(file_parts):
        part_fp = partitions_dir / f"part_{i}.txt"

        with open(part_fp, "w") as f:
            f.write("\n".join(part))

        print(f"Created partition {part_fp}.")


def run_download(
        input_file: str,
        target_dir: pathlib.Path,
        max_files: int,
        aws_config: str
):
    # create downloader
    arxiv_downloader = ArxivDownloader(config_file=aws_config)

    # run download
    arxiv_downloader.run(
        input_file=input_file,
        tgt_dir=target_dir,
        max_files=max_files
    )


def main():
    if not args.local and not args.setup:
        # here we only download the files; this requires that setup has already
        # been run
        run_download(input_file=args.input,
                     target_dir=pathlib.Path(args.target_dir),
                     max_files=args.max_files,
                     aws_config=args.aws_config)
        return

    # create directories
    target_dir = pathlib.Path(args.target_dir)
    partitions_dir = target_dir / 'partitions'

    if args.setup:
        # here we only partition the keys into chunks; no download yet
        partition_keys(partitions_dir=partitions_dir,
                       config_file=args.aws_config,
                       workers=args.workers)
        return

    if args.local:
        partition_keys(partitions_dir=partitions_dir,
                       config_file=args.aws_config,
                       workers=args.workers)

        run_download(input_file=str(partitions_dir / 'part_0.txt'),
                     target_dir=pathlib.Path(args.target_dir),
                     max_files=args.max_files,
                     aws_config=args.aws_config)


if __name__ == '__main__':
    main()
