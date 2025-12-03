#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/test_hnsw_range"

if [[ ! -x "${BIN}" ]]; then
  echo "Binary not found, building..."
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DBUILD_RANGE_BENCHMARKS=ON
  cmake --build "${BUILD_DIR}" --target test_hnsw_range
fi

DATASET="deep"
DATA_SIZE="100000"
DATASET_PATH="${ROOT_DIR}/data/deep10M.bin"
QUERY_PATH="${ROOT_DIR}/data/deep_query.bin"
GT_ROOT="${ROOT_DIR}/groundtruth"

# Create groundtruth directory if it doesn't exist
mkdir -p "${GT_ROOT}"

echo "Running test_hnsw_range..."
"${BIN}" \
  -dataset "${DATASET}" \
  -N "${DATA_SIZE}" \
  -dataset_path "${DATASET_PATH}" \
  -query_path "${QUERY_PATH}" \
  -groundtruth_root "${GT_ROOT}" \
  -query_k 10 \
  -M 16 \
  -ef_construction 100 \
  -ef_search 100 \
  -seed 123


