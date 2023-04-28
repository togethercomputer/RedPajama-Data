# Contamination Analysis

This Folder contains code to run a contamination analysis on the RedPajama dataset, based on n gram overlaps between
testing and training data.

## Algorithm

The methodology is split into two parts:

1. Generate ngrams for the training data
    - This computes Sha1 hashes for each ngram and stores the hash bytes them in binary files.
    - Note that this step requires a lot of disk space (roughly double the size of the training data).
2. Generate ngrams for the testing data and extract the ones which overlap with the training data
    - ngrams of the testing data are computed on the fly and and compared to the hash bytes of the training data.
    - whenever an instance of an ngram is found in the training data, the instance counts as contaminated.

The output of this procedure is a json files for each testing data slice, containing a list
of instance ids contained in the training set.

We run the algorithm for each slice of RP separately in order to analyze how much each training
slice is contaminated.

## Requirements

The code has been tested with python 3.8 and requires the packages specified in `requirements.txt`.

## Usage

[TODO]