#!/bin/bash

set -e

WORKERS=100

DATA_DIR="./data/github/src"
PARTS_DIR="./data/github/partitions"
TARGET_DIR="./data/github/processed"
WORK_DIR="./work"

mkdir -p $DATA_DIR
mkdir -p $TARGET_DIR
mkdir -p $PARTS_DIR
mkdir -p $WORK_DIR
mkdir -p logs/github/download

# set this variable before running the script
GCS_BUCKET=""
if [ -z "$GCS_BUCKET" ]; then
  echo "GCS_BUCKET is not set! Aborting..."
  exit 1
fi

# get a list of all files in the gcp bucket
gsutil ls gs://"${GCS_BUCKET}"/github_*.gz >"$WORK_DIR"/github_file_uris.txt

# Get the total number of files in the current directory
uris_file="$WORK_DIR"/github_file_uris.txt
total_files=$(cat $uris_file | wc -l)

# Calculate the number of files to write to each text file
files_per_text_file=$((total_files / WORKERS))

# Split the file into chunks and write each chunk to a separate file
split --lines=$files_per_text_file $uris_file $PARTS_DIR/uri_parts_

# Rename the chunk files to have a .txt extension
for file in $PARTS_DIR/uri_parts_*; do
  mv "$file" "${file}.txt"
done
