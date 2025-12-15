#!/usr/bin/env bash
set -euo pipefail

# 简单单模型训练封装
# 用法: ./run_train.sh [EXTRA_TRAIN_ARGS...]
# 默认数据路径: ../dataset/ready_v3_t1xTrueDFT_FalseDFT

DATA_PATH="../dataset/ready_v3_t1xTrueDFT_FalseDFT"
TS=$(date +%s)
BASE_LOG="log_single_${TS}"

EXTRA_ARGS=("$@")

echo "[INFO] 单模型训练，数据 ${DATA_PATH}，日志 ${BASE_LOG}"
python train_elementary.py --data_path "${DATA_PATH}" --base_log "${BASE_LOG}" "${EXTRA_ARGS[@]}"
