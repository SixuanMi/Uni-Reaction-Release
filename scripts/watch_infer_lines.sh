#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
Usage:
  ./scripts/watch_infer_lines.sh \
    --input_dir DIR \
    --main_dir VOTE_RUN_DIR \
    --output_dir DIR \
    [infer options]

Required:
  --input_dir DIR          Directory containing line*_aug_stereo.csv files.
  --main_dir DIR           Training output root passed to infer_unlabeled.sh.
  --output_dir DIR         Directory for line*_infer.csv outputs.

Options:
  --interval_seconds N     Scan interval. Default: 1800
  --dim N                  Model hidden dimension. Default: 384
  --num_worker N           DataLoader workers. Default: 15
  --bs N                   Batch size. Default: 2048
  --device ID              GPU id. Default: 0
  --devices IDS            Comma-separated GPU ids for multi-GPU inference, e.g. 0,1,2,3.
                           If provided, this overrides --device.
  --models_per_device_parallel N
                           Model inference processes to run concurrently per GPU. Default: 1
  --reaction_col NAME      Reaction SMILES column. Default: stereo_rsmi
  --fusion_mode MODE       legacy or film. Default: film
  --heads N                Attention heads. Default: 8
  --n_layer N              Model layers. Default: 5
  --negative_slope FLOAT   LeakyReLU negative slope. Default: 0.2
  --seed N                 Random seed. Default: 2025
  --local_heads N          Kept for compatibility. Default: 4
  --once                   Scan once and exit.
  -h, --help               Show this help.

Example:
  nohup ./scripts/watch_infer_lines.sh \
    --input_dir /inspire/qb-ilm/project/chemicalreaction/misixuan-CZXS24220243/github/2daam2d/t1x_depth15_all/raw_outputs/enumerate_steoro \
    --main_dir vote_run_1777029069_depth15_active9_384 \
    --output_dir infer \
    --dim 384 \
    --num_worker 60 \
    --bs 2048 \
    --devices 0,1,2,3 \
    --models_per_device_parallel 2 \
    > infer/watch_infer_lines.log 2>&1 &
EOF
}

INPUT_DIR=""
MAIN_DIR=""
OUTPUT_DIR=""
INTERVAL_SECONDS=1800
DIM=384
NUM_WORKER=15
BS=2048
DEVICE=0
DEVICES=""
MODELS_PER_DEVICE_PARALLEL=1
REACTION_COL="stereo_rsmi"
FUSION_MODE="film"
HEADS=8
N_LAYER=5
NEGATIVE_SLOPE=0.2
SEED=2025
LOCAL_HEADS=4
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir)
      INPUT_DIR="${2:?missing value for --input_dir}"
      shift 2
      ;;
    --main_dir)
      MAIN_DIR="${2:?missing value for --main_dir}"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="${2:?missing value for --output_dir}"
      shift 2
      ;;
    --interval_seconds)
      INTERVAL_SECONDS="${2:?missing value for --interval_seconds}"
      shift 2
      ;;
    --dim)
      DIM="${2:?missing value for --dim}"
      shift 2
      ;;
    --num_worker)
      NUM_WORKER="${2:?missing value for --num_worker}"
      shift 2
      ;;
    --bs)
      BS="${2:?missing value for --bs}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?missing value for --device}"
      shift 2
      ;;
    --devices)
      DEVICES="${2:?missing value for --devices}"
      shift 2
      ;;
    --models_per_device_parallel)
      MODELS_PER_DEVICE_PARALLEL="${2:?missing value for --models_per_device_parallel}"
      shift 2
      ;;
    --reaction_col)
      REACTION_COL="${2:?missing value for --reaction_col}"
      shift 2
      ;;
    --fusion_mode)
      FUSION_MODE="${2:?missing value for --fusion_mode}"
      shift 2
      ;;
    --heads)
      HEADS="${2:?missing value for --heads}"
      shift 2
      ;;
    --n_layer)
      N_LAYER="${2:?missing value for --n_layer}"
      shift 2
      ;;
    --negative_slope)
      NEGATIVE_SLOPE="${2:?missing value for --negative_slope}"
      shift 2
      ;;
    --seed)
      SEED="${2:?missing value for --seed}"
      shift 2
      ;;
    --local_heads)
      LOCAL_HEADS="${2:?missing value for --local_heads}"
      shift 2
      ;;
    --once)
      ONCE=1
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

if [[ -z "${INPUT_DIR}" || -z "${MAIN_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] --input_dir / --main_dir / --output_dir are required" >&2
  usage
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] input_dir does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

infer_one_file() {
  local input_path="$1"
  local base line_id output_path tmp_output lock_dir done_marker

  base=$(basename "${input_path}")
  line_id="${base%_aug_stereo.csv}"
  output_path="${OUTPUT_DIR%/}/${line_id}_infer.csv"
  tmp_output="${OUTPUT_DIR%/}/.${line_id}_infer.tmp.csv"
  lock_dir="${OUTPUT_DIR%/}/.${line_id}.lock"
  done_marker="${output_path}.done"

  if [[ -s "${output_path}" || -f "${done_marker}" ]]; then
    log "[SKIP] ${base} -> ${output_path} already exists"
    return 0
  fi

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    log "[SKIP] ${base} is locked by another process"
    return 0
  fi
  trap 'rm -rf "${lock_dir}"' RETURN

  rm -f "${tmp_output}"
  log "[RUN ] ${base} -> ${output_path}"
  local infer_args=(
    --input "${input_path}"
    --main_dir "${MAIN_DIR}"
    --output "${tmp_output}"
    --dim "${DIM}"
    --num_worker "${NUM_WORKER}"
    --bs "${BS}"
    --reaction_col "${REACTION_COL}"
    --fusion_mode "${FUSION_MODE}"
    --heads "${HEADS}"
    --n_layer "${N_LAYER}"
    --negative_slope "${NEGATIVE_SLOPE}"
    --seed "${SEED}"
    --local_heads "${LOCAL_HEADS}"
    --models_per_device_parallel "${MODELS_PER_DEVICE_PARALLEL}"
  )
  if [[ -n "${DEVICES}" ]]; then
    infer_args+=(--devices "${DEVICES}")
  else
    infer_args+=(--device "${DEVICE}")
  fi

  if "${SCRIPT_DIR}/infer_unlabeled.sh" "${infer_args[@]}"; then
    mv "${tmp_output}" "${output_path}"
    date '+%Y-%m-%d %H:%M:%S' > "${done_marker}"
    log "[DONE] ${base} -> ${output_path}"
  else
    local status=$?
    rm -f "${tmp_output}"
    log "[FAIL] ${base}, exit=${status}"
    return "${status}"
  fi

  rm -rf "${lock_dir}"
  trap - RETURN
}

scan_once() {
  local found=0
  shopt -s nullglob
  for input_path in "${INPUT_DIR%/}"/line*_aug_stereo.csv; do
    found=1
    infer_one_file "${input_path}" || true
  done
  shopt -u nullglob

  if [[ "${found}" -eq 0 ]]; then
    log "[INFO] No line*_aug_stereo.csv found in ${INPUT_DIR}"
  fi
}

log "[INFO] Watching ${INPUT_DIR}"
log "[INFO] Output dir: ${OUTPUT_DIR}"
log "[INFO] Interval: ${INTERVAL_SECONDS}s"

while true; do
  scan_once
  if [[ "${ONCE}" -eq 1 ]]; then
    break
  fi
  log "[INFO] Sleeping ${INTERVAL_SECONDS}s"
  sleep "${INTERVAL_SECONDS}"
done
