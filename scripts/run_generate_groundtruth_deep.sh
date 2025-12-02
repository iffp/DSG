#!/usr/bin/env bash
set -euo pipefail

# 2025-12-01 Zhencan Peng: Iterate deep/wikipedia/yt8m datasets with custom paths.
# Set ROOT_DIR to the parent directory of the location of this script.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/apps/generate_groundtruth"

if [[ ! -x "${BIN}" ]]; then
  echo "Binary ${BIN} not found. Build it first with: cmake -S ${ROOT_DIR} -B ${BUILD_DIR} && cmake --build ${BUILD_DIR} --target generate_groundtruth" >&2
  exit 1
fi

GROUND_DIR="${GROUND_DIR:-${ROOT_DIR}/groundtruth/static}"
declare -A DEFAULT_DATASET_PATHS=(
  ["deep"]="${ROOT_DIR}/data/deep10M.bin"
  ["wikipedia"]="${ROOT_DIR}/data/wiki_image_embedding.bin"
  ["yt8m"]="${ROOT_DIR}/data/yt8m_sorted_by_timestamp_video_embedding_1M.bin"
)
declare -A DEFAULT_QUERY_PATHS=(
  ["deep"]="${ROOT_DIR}/data/deep_query.bin"
  ["wikipedia"]="${ROOT_DIR}/data/wiki_image_query.bin"
  ["yt8m"]="${ROOT_DIR}/data/yt8m_video_query_10k.bin"
)

datasets=("yt8m") #"deep" "wikipedia" 

for dataset in "${datasets[@]}"; do
  dataset_path="${DEFAULT_DATASET_PATHS[${dataset}]:-}"
  query_path="${DEFAULT_QUERY_PATHS[${dataset}]:-}"

  if [[ -z "${dataset_path}" || -z "${query_path}" ]]; then
    echo "Dataset or query path missing for ${dataset}." >&2
    exit 1
  fi

  echo "Running groundtruth generation for dataset ${dataset}..."
  "${BIN}" \
    -dataset "${dataset}" \
    -N 100000 \
    -dataset_path "${dataset_path}" \
    -query_path "${query_path}" \
    -groundtruth_root "${GROUND_DIR}"
done

