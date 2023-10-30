#!/bin/bash

set -e
trap cleanup_on_error ERR SIGINT SIGTERM

cleanup_on_error() {
  echo "Error: $0:$LINENO: command \`$BASH_COMMAND\` failed with exit code $?"
  exit 1
}

help() {
  echo "Usage: apptainer_run_quality_signals.sh [ -c | --config ] [ -d | --dump_id ]"
  exit 2
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  -c | --config)
    CONFIG_FILE="$2"
    shift 2
    ;;
  -d | --dump_id)
    DUMP_ID="$2"
    shift 2
    ;;
  -l | --listings)
    LISTINGS="$2"
    shift 2
    ;;
  -h | --help)
    help
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "Invalid option: -$1"
    help
    ;;
  esac
done

# make environment variables available to downstream scripts
set -a
# shellcheck source=configs/base.conf
. "$CONFIG_FILE"
set +a

if [ -z "${MAX_DOCS}" ]; then
  MAX_DOCS=-1
fi

ARTIFACTS_ARCHIVE="${DATA_ROOT%/}/artifacts-${ARTIFACTS_ID}.tar.gz"

if [ ! -d "${DATA_ROOT%/}/artifacts-${ARTIFACTS_ID}" ]; then
  # download artifacts from bucket
  echo "Downloading artifacts from ${INPUT_BASE_URI%/}/artifacts-${ARTIFACTS_ID}.tar.gz"
  s5cmd --profile "$S3_PROFILE" --endpoint-url "$S3_ENDPOINT_URL" \
    cp "${S3_BUCKET%/}/artifacts/artifacts-${ARTIFACTS_ID}.tar.gz" "${ARTIFACTS_ARCHIVE}"

  # extract artifacts
  mkdir -p "${DATA_ROOT%/}/artifacts-${ARTIFACTS_ID}"
  echo "Extracting artifacts to ${DATA_ROOT%/}/artifacts-${ARTIFACTS_ID}"
  tar -xzf "${ARTIFACTS_ARCHIVE}" -C "${DATA_ROOT%/}/artifacts-${ARTIFACTS_ID}"
  rm "${ARTIFACTS_ARCHIVE}"
else
  echo "Artifacts already exist at ${DATA_ROOT%/}/artifacts-${ARTIFACTS_ID}; skipping download."
fi


# run pipeline
ARTIFACTS_DIR="${DATA_ROOT%/}/artifacts-${ARTIFACTS_ID}"

if [ -z "${LISTINGS}" ]; then
  LISTINGS="${ARTIFACTS_DIR%/}/listings/listings-${DUMP_ID}.txt"
fi

apptainer cache clean -f
apptainer run \
  --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  "docker://docker.io/${DOCKER_REPO}:amd64" \
  python3 /usr/app/src/pipeline.py \
  --input "${LISTINGS}" \
  --input_base_uri "${INPUT_BASE_URI}" \
  --output_base_uri "${OUTPUT_BASE_URI}" \
  --cc_snapshot_id "${DUMP_ID}" \
  --artifacts_dir "${ARTIFACTS_DIR}" \
  --dsir_buckets "${DSIR_FEATURE_DIM}" \
  --max_docs "${MAX_DOCS}" \
  --inputs_per_process "${INPUTS_PER_PROCESS}" \
  --langs "${LANGUAGES[@]}" \
  --endpoint_url "${S3_ENDPOINT_URL}" \
  --minhash_ngram_size "${MINHASH_NGRAM_SIZE}" \
  --minhash_num_permutations "${MINHASH_NUM_PERMUTATIONS}" \
  --minhash_similarities "${MINHASH_SIMILARITIES[@]}" \
  --filename_keep_patterns "${FILENAME_KEEP_PATTERNS[@]}"
