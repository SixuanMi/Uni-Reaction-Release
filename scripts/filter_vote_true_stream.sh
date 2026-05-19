#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
Usage:
  Single file:
    ./scripts/filter_vote_true_stream.sh \
      --input infer/line1_infer.csv \
      --output infer_true/line1_true.csv

  Directory batch:
    ./scripts/filter_vote_true_stream.sh \
      --input_dir infer \
      --output_dir infer_true

Options:
  --input PATH             One inference CSV to filter.
  --output PATH            Output CSV for --input mode.
  --input_dir DIR          Directory containing inference CSV files.
  --output_dir DIR         Output directory for --input_dir mode.
  --pattern GLOB           Input filename glob under input_dir. Default: line*_infer.csv
  --chunksize N            Rows per pandas chunk. Default: 100000
  --cls_threshold FLOAT    Classification probability threshold. Default: 0.5
  --comparison gt|ge       gt means mean cls_prob > threshold; ge means >=. Default: gt
  --expected_models N      Fail unless this many model*_cls_prob columns are found. Default: 8
  --interval_seconds N     Scan interval when --watch is set. Default: 1800
  --watch                  Keep scanning input_dir and process newly finished files.
  --once                   Alias for default one-shot directory scan.
  -h, --help               Show this help.

Output naming in directory mode:
  line1_infer.csv -> line1_true.csv
  other.csv       -> other_true.csv

Example:
  nohup ./scripts/filter_vote_true_stream.sh \
    --input_dir infer \
    --output_dir infer_true \
    --chunksize 100000 \
    --comparison gt \
    --cls_threshold 0.5 \
    --watch \
    > infer_true/filter_vote_true.log 2>&1 &
EOF
}

INPUT=""
OUTPUT=""
INPUT_DIR=""
OUTPUT_DIR=""
PATTERN="line*_infer.csv"
CHUNKSIZE=100000
CLS_THRESHOLD=0.5
COMPARISON="gt"
EXPECTED_MODELS=8
INTERVAL_SECONDS=1800
WATCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="${2:?missing value for --input}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:?missing value for --output}"
      shift 2
      ;;
    --input_dir)
      INPUT_DIR="${2:?missing value for --input_dir}"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="${2:?missing value for --output_dir}"
      shift 2
      ;;
    --pattern)
      PATTERN="${2:?missing value for --pattern}"
      shift 2
      ;;
    --chunksize)
      CHUNKSIZE="${2:?missing value for --chunksize}"
      shift 2
      ;;
    --cls_threshold)
      CLS_THRESHOLD="${2:?missing value for --cls_threshold}"
      shift 2
      ;;
    --comparison)
      COMPARISON="${2:?missing value for --comparison}"
      shift 2
      ;;
    --expected_models)
      EXPECTED_MODELS="${2:?missing value for --expected_models}"
      shift 2
      ;;
    --interval_seconds)
      INTERVAL_SECONDS="${2:?missing value for --interval_seconds}"
      shift 2
      ;;
    --watch)
      WATCH=1
      shift
      ;;
    --once)
      WATCH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${COMPARISON}" != "gt" && "${COMPARISON}" != "ge" ]]; then
  echo "[ERROR] --comparison must be gt or ge" >&2
  exit 1
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

derive_output_path() {
  local input_path="$1"
  local base stem out_stem
  base=$(basename "${input_path}")
  stem="${base%.csv}"
  if [[ "${stem}" == *_infer ]]; then
    out_stem="${stem%_infer}_true"
  else
    out_stem="${stem}_true"
  fi
  echo "${OUTPUT_DIR%/}/${out_stem}.csv"
}

filter_one_file() {
  local input_path="$1"
  local output_path="$2"
  local lock_dir done_marker

  if [[ ! -s "${input_path}" ]]; then
    log "[SKIP] missing or empty input: ${input_path}"
    return 0
  fi

  lock_dir="${output_path}.lock"
  done_marker="${output_path}.done"

  if [[ -s "${output_path}" || -f "${done_marker}" ]]; then
    log "[SKIP] ${input_path} -> ${output_path} already exists"
    return 0
  fi

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    log "[SKIP] ${input_path} is locked by another process"
    return 0
  fi
  trap 'rm -rf "${lock_dir}"' RETURN

  log "[RUN ] ${input_path} -> ${output_path}"
  if python "${SCRIPT_DIR}/_filter_vote_true_stream.py" \
    --input "${input_path}" \
    --output "${output_path}" \
    --chunksize "${CHUNKSIZE}" \
    --cls_threshold "${CLS_THRESHOLD}" \
    --comparison "${COMPARISON}" \
    --expected_models "${EXPECTED_MODELS}"; then
    date '+%Y-%m-%d %H:%M:%S' > "${done_marker}"
    log "[DONE] ${input_path} -> ${output_path}"
  else
    local status=$?
    log "[FAIL] ${input_path}, exit=${status}"
    return "${status}"
  fi

  rm -rf "${lock_dir}"
  trap - RETURN
}

scan_once() {
  local found=0
  shopt -s nullglob
  for input_path in "${INPUT_DIR%/}"/${PATTERN}; do
    found=1
    filter_one_file "${input_path}" "$(derive_output_path "${input_path}")" || true
  done
  shopt -u nullglob

  if [[ "${found}" -eq 0 ]]; then
    log "[INFO] No files matched ${INPUT_DIR%/}/${PATTERN}"
  fi
}

if [[ -n "${INPUT}" || -n "${OUTPUT}" ]]; then
  if [[ -z "${INPUT}" || -z "${OUTPUT}" ]]; then
    echo "[ERROR] --input and --output must be provided together" >&2
    usage
    exit 1
  fi
  mkdir -p "$(dirname "${OUTPUT}")"
  filter_one_file "${INPUT}" "${OUTPUT}"
  exit 0
fi

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] Provide either --input/--output or --input_dir/--output_dir" >&2
  usage
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] input_dir does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

log "[INFO] Input dir: ${INPUT_DIR}"
log "[INFO] Output dir: ${OUTPUT_DIR}"
log "[INFO] Pattern: ${PATTERN}"
log "[INFO] Comparison: mean cls_prob ${COMPARISON} ${CLS_THRESHOLD}"
log "[INFO] Chunksize: ${CHUNKSIZE}"

while true; do
  scan_once
  if [[ "${WATCH}" -eq 0 ]]; then
    break
  fi
  log "[INFO] Sleeping ${INTERVAL_SECONDS}s"
  sleep "${INTERVAL_SECONDS}"
done
