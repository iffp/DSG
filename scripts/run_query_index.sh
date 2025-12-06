#!/usr/bin/env bash
set -euo pipefail

# 2025-12-02 Zhencan Peng: run query_index with default deep-100k parameters.

usage() {
  cat <<'EOF'
Usage: run_query_index.sh [SEARCH_EF]

Optional SEARCH_EF overrides the search ef passed to query_index.
EOF
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BUILD_DIR}/apps/query_index"

configure_and_build() {
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" --target query_index
}

if [[ ! -x "${BIN}" ]]; then
  echo "[DSG] query_index binary not found, building..."
  configure_and_build
fi

SEARCH_EF=""
if [[ $# -gt 1 ]]; then
  usage
elif [[ $# -eq 1 ]]; then
  SEARCH_EF="$1"
fi

DATASET="deep"
DATA_SIZE="100000"
DATASET_PATH="${ROOT_DIR}/data/deep10M.bin"
QUERY_PATH="${ROOT_DIR}/data/deep_query.bin"
INDEX_PATH="${ROOT_DIR}/index/${DATASET}_N${DATA_SIZE}.index"
GROUND_ROOT="${ROOT_DIR}/groundtruth/static"
QUERY_NUM="1000"
QUERY_K="10"

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "[DSG] Index file ${INDEX_PATH} not found." >&2
  exit 1
fi

CMD_ARGS=(
  -dataset "${DATASET}"
  -N "${DATA_SIZE}"
  -dataset_path "${DATASET_PATH}"
  -query_path "${QUERY_PATH}"
  -index_path "${INDEX_PATH}"
  -groundtruth_root "${GROUND_ROOT}"
  -query_num "${QUERY_NUM}"
  -query_k "${QUERY_K}"
)

if [[ -n "${SEARCH_EF}" ]]; then
  CMD_ARGS+=(-search_ef "${SEARCH_EF}")
fi

"${BIN}" "${CMD_ARGS[@]}"



