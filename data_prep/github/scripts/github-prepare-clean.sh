#!/bin/bash

set -e

WORKERS=100

export DATA_DIR="./data/github/src"
export PARTS_DIR="./data/github/processed/partitions"
export TARGET_DIR="./data/github/processed"
export WORK_DIR="./work"

mkdir -p $TARGET_DIR
mkdir -p $PARTS_DIR
mkdir -p logs/github/cleaning

# get a list of all files in the data directory
ls $DATA_DIR/github_*.gz >"$WORK_DIR"/github_files.txt

# Get the total number of files
all_files="$WORK_DIR"/github_files.txt
total_files=$(cat $all_files | wc -l)

# Calculate the number of files to write to each text file
files_per_chunk=$((total_files / WORKERS))

# Split the file into chunks and write each chunk to a separate file
split --lines=$files_per_chunk $all_files $PARTS_DIR/chunk_

# Rename the chunk files
for file in $PARTS_DIR/chunk_*; do
  mv "$file" "${file}.txt"
done
