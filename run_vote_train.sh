#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_vote_train.sh DATA_PATH [EXTRA_TRAIN_ARGS...]
# DATA_PATH: 原始数据目录（包含 train/val/test 或 combined.csv）
# 环境变量: N_FOLDS(默认5), TEST_SIZE(默认0.1), SEED(默认2025)

DATA_PATH=${1:?please provide data path}
shift || true

N_FOLDS=${N_FOLDS:-5}
TEST_SIZE=${TEST_SIZE:-0.1}
SEED=${SEED:-2025}

TS=$(date +%s)
BASE_DIR="vote_run_${TS}"
SPLIT_DIR="${BASE_DIR}/folds"
LOG_ROOT="${BASE_DIR}/logs"

echo "[INFO] 生成分层拆分: 全局测试占比 ${TEST_SIZE}, 折数 ${N_FOLDS}, 输出 ${SPLIT_DIR}"
python train_elementary_vote.py \
  --data_path "${DATA_PATH}" \
  --output_dir "${SPLIT_DIR}" \
  --n_folds "${N_FOLDS}" \
  --test_size "${TEST_SIZE}" \
  --seed "${SEED}"

# 依次训练每个折的模型，日志/模型目录归档在 BASE_DIR 下
EXTRA_ARGS=("$@")
for i in $(seq 1 "$N_FOLDS"); do
  FOLD_DIR="${SPLIT_DIR}/fold_${i}"
  LOG_DIR="${LOG_ROOT}/fold_${i}"
  echo "========================================="
  echo "[INFO] 训练折 ${i}/${N_FOLDS}，数据 ${FOLD_DIR}，日志根目录 ${LOG_DIR}"
  python train_elementary.py --data_path "${FOLD_DIR}" --base_log "${LOG_DIR}" "${EXTRA_ARGS[@]}"
done
