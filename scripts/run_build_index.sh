#!/usr/bin/env bash
set -euo pipefail

# 2025-12-02 Zhencan Peng: convenience wrapper to build and run build_index with default parameters.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/apps/build_index"

configure_and_build() {
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" --target build_index
}

if [[ ! -x "${BIN}" ]]; then
  echo "[DSG] build_index binary not found, building..."
  configure_and_build
fi

DATASET="deep"
DATA_SIZE="100000"
DATASET_PATH="${ROOT_DIR}/data/deep10M.bin"
QUERY_PATH="${ROOT_DIR}/data/deep_query.bin"
INDEX_PATH="${ROOT_DIR}/index/${DATASET}_N${DATA_SIZE}.index"
INDEX_K="16"
EF_CONSTRUCTION="100"
EF_MAX="200"
ALPHA="1.0"

mkdir -p "$(dirname "${INDEX_PATH}")"

echo "[DSG] Running build_index..."
"${BIN}" \
  -dataset "${DATASET}" \
  -N "${DATA_SIZE}" \
  -dataset_path "${DATASET_PATH}" \
  -query_path "${QUERY_PATH}" \
  -index_path "${INDEX_PATH}" \
  -k "${INDEX_K}" \
  -ef_construction "${EF_CONSTRUCTION}" \
  -ef_max "${EF_MAX}" \
  -alpha "${ALPHA}"

