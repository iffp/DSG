#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/test_hnsw_range"
SRC_DIR="${ROOT_DIR}/src"
INCLUDE_DIR="${ROOT_DIR}/include"

# Recompile locally with g++ if requested or missing.
if [[ ! -x "${BIN}" ]]; then
  mkdir -p "${BUILD_DIR}"
  echo "Compiling test_hnsw_range with g++ (standalone)..."
  g++ -std=c++17 -O3 -march=native -fopenmp -msse4.2 \
    -I"${INCLUDE_DIR}" \
    -I"${INCLUDE_DIR}/utils" \
    -I"${ROOT_DIR}/src" \
    "${ROOT_DIR}/tests/test_hnsw_range.cpp" \
    "${SRC_DIR}/utils/data_wrapper.cc" \
    "${SRC_DIR}/utils/reader.cc" \
    "${SRC_DIR}/utils/utils.cc" \
    -o "${BIN}"
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


