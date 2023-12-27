#!/bin/bash

set -e
trap cleanup_on_error ERR SIGINT SIGTERM

cleanup_on_error() {
  echo "Error: $0:$LINENO: command \`$BASH_COMMAND\` failed with exit code $?"
  exit 1
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --config)
    CONFIG_FILE="$2"
    shift
    shift
    ;;
  --input_base_uri)
    INPUT_BASE_URI="$2"
    shift
    shift
    ;;
  --output_dir)
    OUTPUT_DIR="$2"
    shift
    shift
    ;;
  --similarity)
    SIMILARITY="$2"
    shift
    shift
    ;;
  --listings)
    LISTINGS="$2"
    shift
    shift
    ;;
  --max_docs)
    MAX_DOCS="$2"
    shift
    shift
    ;;
  *)
    echo "Invalid option: -$OPTARG" >&2
    ;;
  esac
done

# make environment variables available to downstream scripts
set -a
# shellcheck source=configs/base.conf
. "$CONFIG_FILE"
set +a

# run pipeline
apptainer run --memory 480g "${DOCKER_REPO}" \
  python3 src/run_lsh.py \
  --listings "${LISTINGS}" \
  --input_base_uri "${INPUT_BASE_URI}" \
  --output_dir "${OUTPUT_DIR}" \
  --similarity "${SIMILARITY}" \
  --num_perm "${MINHASH_NUM_PERMUTATIONS}" \
  --max_docs ${MAX_DOCS}
