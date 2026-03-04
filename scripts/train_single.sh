#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./scripts/train_single.sh DATA_PATH [EXTRA_TRAIN_ARGS...]
#   ./scripts/train_single.sh --data_path DATA_PATH [EXTRA_TRAIN_ARGS...]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ "${1:-}" == "--data_path" ]]; then
  DATA_PATH=${2:?please provide data path}
  shift 2 || true
else
  DATA_PATH=${1:?please provide data path}
  shift || true
fi

TS=$(date +%s)
BASE_LOG=${BASE_LOG:-"log_single_${TS}"}
EXTRA_ARGS=("$@")

echo "[INFO] 单模型训练，数据 ${DATA_PATH}，日志根目录 ${BASE_LOG}"
python "${ROOT_DIR}/train_elementary.py" \
  --data_path "${DATA_PATH}" \
  --base_log "${BASE_LOG}" \
  "${EXTRA_ARGS[@]}"
