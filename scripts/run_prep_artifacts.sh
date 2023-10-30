#!/bin/bash

set -e
trap cleanup_on_error ERR SIGINT SIGTERM

function cleanup_on_error() {
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
  --listings)
    RAW_LISTINGS_FILE="$2"
    shift
    shift
    ;;
  --max_workers)
    MAX_WORKERS="$2"
    shift
    shift
    ;;
  *)
    echo "Invalid option: -$OPTARG" >&2
    ;;
  esac
done

set -a
# shellcheck source=configs/base.conf
. "$CONFIG_FILE"
set +a

# create random uuid if not provided
RUN_ID=$(openssl rand -hex 4)
echo "Created run id: $RUN_ID"

ARTIFACTS_DIR="${DATA_ROOT%/}/artifacts-${RUN_ID}"
LISTINGS_DIR="${ARTIFACTS_DIR%/}/listings"
LISTINGS_FILE="${LISTINGS_DIR%/}/listings.txt"
mkdir -p "${LISTINGS_DIR%/}"

# write id to file if it doesn't exist
if [ ! -f "${ARTIFACTS_DIR%/}/_RUN_ID" ]; then
  echo "Writing run id to file ${ARTIFACTS_DIR%/}/_RUN_ID"
  echo "$RUN_ID" >"${ARTIFACTS_DIR%/}/_RUN_ID"
fi

# fetch listings from s3 bucket if the listings file does not exist
if [ ! -f "${RAW_LISTINGS_FILE}" ]; then
  echo "__FETCH_LISTINGS_START__  @ $(date) Fetching listings from s3 bucket..."
  s5cmd --profile "$S3_PROFILE" --endpoint-url "$S3_ENDPOINT_URL" \
    ls "${S3_BUCKET%/}${S3_CCNET_PREFIX%/}/*" |
    grep "\.json\.gz$" | awk '{print $NF}' >"${LISTINGS_FILE}"
  echo "__FETCH_LISTINGS_END__  @ $(date) Done tetching listings from s3 bucket."
else
  cp "${RAW_LISTINGS_FILE}" "${LISTINGS_FILE}"
  echo "copied listings file from ${RAW_LISTINGS_FILE} to ${LISTINGS_FILE}"
fi

# create a listings for each snapshot id if
for snapshot_id in "${CC_SNAPSHOT_IDS[@]}"; do
  if grep "${snapshot_id}" "${LISTINGS_DIR%/}/listings.txt" >/dev/null 2>&1; then
    grep "${snapshot_id}" "${LISTINGS_DIR%/}/listings.txt" >"${LISTINGS_DIR%/}/listings-${snapshot_id}.txt"
    echo "__SNAPSHOT_LISTINGS_SUCCESS__ $snapshot_id"
  else
    echo "__SNAPSHOT_LISTINGS_FAIL__ $snapshot_id"
  fi
done

num_listings=$(wc -l <"${ARTIFACTS_DIR%/}/listings/listings.txt")
echo "Toal number of listings: $num_listings"

# copy config to artifacts dir
cp "$CONFIG_FILE" "${ARTIFACTS_DIR%/}/config.conf"

# Reset artifacts dir on docker mounted volume
ARTIFACTS_DIR="${DOCKER_MNT_DIR%/}/artifacts-${RUN_ID}"
for lang in "${LANGUAGES[@]}"; do
  echo "__LANG_PREP_START__ ${lang} @ $(date)"
  docker run --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    -v "${DATA_ROOT%/}":"${DOCKER_MNT_DIR%/}" -t "${DOCKER_REPO}" \
    python3 src/prep_artifacts.py \
    --artifacts_dir "${ARTIFACTS_DIR%/}" \
    --cc_input "${ARTIFACTS_DIR%/}/listings/listings.txt" \
    --cc_input_base_uri "${S3_BUCKET%/}${S3_CCNET_PREFIX%/}" \
    --cache_dir "${DOCKER_MNT_DIR%/}/.hf_cache" \
    --lang "${lang}" \
    --max_workers "${MAX_WORKERS}" \
    --endpoint_url "$DOCKER_S3_ENDPOINT_URL" \
    --dsir_num_samples "${DSIR_NUM_SAMPLES}" \
    --dsir_feature_dim "${DSIR_FEATURE_DIM}" \
    --classifiers_num_samples "${CLASSIFIERS_NUM_SAMPLES}" \
    --max_paragraphs_per_book_sample "${MAX_PARAGRAPHS_PER_BOOK_SAMPLE}" \
    --max_samples_per_book "${MAX_SAMPLES_PER_BOOK}"
  echo "__LANG_PREP_END__ ${lang} @ $(date)"
done

echo "__UPDATE_CONENTLISTS_START__ @ $(date)"
docker run -v "${DATA_ROOT%/}":"${DOCKER_MNT_DIR%/}" -t "${DOCKER_REPO}" \
  python3 src/artifacts/update_resources.py \
  --langs "${LANGUAGES[@]}" \
  --artifacts_dir "${ARTIFACTS_DIR%/}" \
  --block_categories "${DOMAIN_BLACKLIST_CATEGORIES[@]}"

echo "__UPDATE_CONENTLISTS_END__ @ $(date)"

# package artifacts
echo "__PACKAGE_ARTIFACTS_START__ @ $(date)"
ARTIFACTS_DIR="${DATA_ROOT%/}/artifacts-${RUN_ID}"
EXPORT_ARTIFACTS="${DATA_ROOT%/}/_EXPORT_artifacts-${RUN_ID}"
mkdir -p "${EXPORT_ARTIFACTS%/}"

# copy wikiref model to artifacts dir
cp "${DATA_ROOT%/}/wikiref-models/en/en-model.bin" "${ARTIFACTS_DIR%/}/classifiers/en/wikiref.model.bin"

# move artifacts to export
cp -r "${ARTIFACTS_DIR%/}/dsir" "${EXPORT_ARTIFACTS%/}/"
cp -r "${ARTIFACTS_DIR%/}/classifiers" "${EXPORT_ARTIFACTS%/}/"
cp -r "${ARTIFACTS_DIR%/}/bad_words" "${EXPORT_ARTIFACTS%/}/"
cp -r "${ARTIFACTS_DIR%/}/bad_urls" "${EXPORT_ARTIFACTS%/}/"
cp -r "${ARTIFACTS_DIR%/}/listings" "${EXPORT_ARTIFACTS%/}/"
cp -r "${ARTIFACTS_DIR%/}/_RUN_ID" "${EXPORT_ARTIFACTS%/}/"
cp -r "${ARTIFACTS_DIR%/}/logs" "${EXPORT_ARTIFACTS%/}/"

# package artifacts
tar -czf "${EXPORT_ARTIFACTS%/}.tar.gz" -C "${EXPORT_ARTIFACTS%/}" .
echo "__PACKAGE_ARTIFACTS_END__ @ $(date)"
