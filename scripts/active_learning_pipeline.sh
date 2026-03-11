#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

usage() {
  cat <<'EOF'
用法:
  ./scripts/active_learning_pipeline.sh \
    --input INPUT_CSV \
    --main_dir VOTE_RUN_DIR \
    --output_dir OUTPUT_DIR \
    [--devices 0,1,2,3] \
    [--reaction_col stereo_rsmi] \
    [--fusion_mode film] \
    [--top_n 100] \
    [--uncertainty_metric entropy|bald] \
    [--reg_uncertainty_metric std|binned_z] \
    [--infer_csv infer_unlabeled.csv] \
    [--select_csv selected_top100.csv] \
    [--dim 192] [--heads 8] [--n_layer 5] \
    [--negative_slope 0.2] [--bs 128] [--num_worker 8] \
    [--device 0] [--seed 2025] [--local_heads 4]
EOF
}

INPUT=""
MAIN_DIR=""
OUTPUT_DIR=""
DEVICES=""
REACTION_COL="stereo_rsmi"
FUSION_MODE="film"
TOP_N="100"
UNCERTAINTY_METRIC="entropy"
REG_UNCERTAINTY_METRIC="binned_z"
INFER_CSV="infer_unlabeled.csv"
SELECT_CSV=""

DIM="192"
HEADS="8"
N_LAYER="5"
NEGATIVE_SLOPE="0.2"
BS="128"
NUM_WORKER="8"
DEVICE="0"
SEED="2025"
LOCAL_HEADS="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="${2:?missing value for --input}"
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
    --devices)
      DEVICES="${2:?missing value for --devices}"
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
    --top_n)
      TOP_N="${2:?missing value for --top_n}"
      shift 2
      ;;
    --uncertainty_metric)
      UNCERTAINTY_METRIC="${2:?missing value for --uncertainty_metric}"
      shift 2
      ;;
    --reg_uncertainty_metric)
      REG_UNCERTAINTY_METRIC="${2:?missing value for --reg_uncertainty_metric}"
      shift 2
      ;;
    --infer_csv)
      INFER_CSV="${2:?missing value for --infer_csv}"
      shift 2
      ;;
    --select_csv)
      SELECT_CSV="${2:?missing value for --select_csv}"
      shift 2
      ;;
    --dim)
      DIM="${2:?missing value for --dim}"
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
    --bs)
      BS="${2:?missing value for --bs}"
      shift 2
      ;;
    --num_worker)
      NUM_WORKER="${2:?missing value for --num_worker}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?missing value for --device}"
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT}" || -z "${MAIN_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] --input / --main_dir / --output_dir 为必填参数" >&2
  usage
  exit 1
fi

if [[ -z "${SELECT_CSV}" ]]; then
  SELECT_CSV="selected_top${TOP_N}.csv"
fi

mkdir -p "${OUTPUT_DIR}"

INFER_OUT="${OUTPUT_DIR%/}/${INFER_CSV}"
SELECT_OUT="${OUTPUT_DIR%/}/${SELECT_CSV}"

echo "[INFO] Step 1/2: Infer unlabeled reactions"
INFER_ARGS=(
  --input "${INPUT}"
  --main_dir "${MAIN_DIR}"
  --output "${INFER_OUT}"
  --reaction_col "${REACTION_COL}"
  --fusion_mode "${FUSION_MODE}"
  --dim "${DIM}"
  --heads "${HEADS}"
  --n_layer "${N_LAYER}"
  --negative_slope "${NEGATIVE_SLOPE}"
  --bs "${BS}"
  --num_worker "${NUM_WORKER}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --local_heads "${LOCAL_HEADS}"
)
if [[ -n "${DEVICES}" ]]; then
  INFER_ARGS+=(--devices "${DEVICES}")
fi
"${SCRIPT_DIR}/infer_unlabeled.sh" "${INFER_ARGS[@]}"

echo "[INFO] Step 2/2: Active selection"
"${SCRIPT_DIR}/active_select.sh" \
  --input "${INFER_OUT}" \
  --top_n "${TOP_N}" \
  --output "${SELECT_OUT}" \
  --uncertainty_metric "${UNCERTAINTY_METRIC}" \
  --reg_uncertainty_metric "${REG_UNCERTAINTY_METRIC}"

echo "[INFO] Pipeline finished"
echo "[INFO] Infer output : ${INFER_OUT}"
echo "[INFO] Select output: ${SELECT_OUT}"
