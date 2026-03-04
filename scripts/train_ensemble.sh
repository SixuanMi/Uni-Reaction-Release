#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./scripts/train_ensemble.sh DATA_PATH [EXTRA_TRAIN_ARGS...]
#   ./scripts/train_ensemble.sh --data_path DATA_PATH [EXTRA_TRAIN_ARGS...]
#
# 环境变量:
#   N_FOLDS(默认5), TEST_SIZE(默认0.1), SEED(默认2025)
#   PARALLEL_JOBS(默认1): 同时训练的 fold 数量
#   GPU_IDS(默认"0"): 逗号分隔 GPU 编号，按 fold 轮询分配

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ "${1:-}" == "--data_path" ]]; then
  DATA_PATH=${2:?please provide data path}
  shift 2 || true
else
  DATA_PATH=${1:?please provide data path}
  shift || true
fi

N_FOLDS=${N_FOLDS:-5}
TEST_SIZE=${TEST_SIZE:-0.1}
SEED=${SEED:-2025}
PARALLEL_JOBS=${PARALLEL_JOBS:-1}
GPU_IDS=${GPU_IDS:-0}

if (( PARALLEL_JOBS < 1 )); then
  echo "[ERROR] PARALLEL_JOBS 必须 >= 1，当前为 ${PARALLEL_JOBS}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
if (( ${#GPU_ARR[@]} == 0 )); then
  echo "[ERROR] GPU_IDS 不能为空" >&2
  exit 1
fi

TS=$(date +%s)
BASE_DIR=${BASE_DIR:-"vote_run_${TS}"}
SPLIT_DIR="${BASE_DIR}/folds"
LOG_ROOT="${BASE_DIR}/logs"

echo "[INFO] 生成分层拆分: 全局测试占比 ${TEST_SIZE}, 折数 ${N_FOLDS}, 输出 ${SPLIT_DIR}"
python "${ROOT_DIR}/prepare_joint_folds.py" \
  --data_path "${DATA_PATH}" \
  --output_dir "${SPLIT_DIR}" \
  --n_folds "${N_FOLDS}" \
  --test_size "${TEST_SIZE}" \
  --seed "${SEED}"

EXTRA_ARGS=("$@")
PIDS=()
PIDS_INFO=()

echo "[INFO] 开始训练：并发数 ${PARALLEL_JOBS}, GPU列表 ${GPU_IDS}"
for i in $(seq 1 "${N_FOLDS}"); do
  FOLD_DIR="${SPLIT_DIR}/fold_${i}"
  LOG_DIR="${LOG_ROOT}/fold_${i}"
  gpu_idx=$(( (i - 1) % ${#GPU_ARR[@]} ))
  gpu_id="${GPU_ARR[$gpu_idx]}"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/train.log"

  while (( $(jobs -pr | wc -l) >= PARALLEL_JOBS )); do
    sleep 2
  done

  echo "========================================="
  echo "[INFO] 启动折 ${i}/${N_FOLDS}，GPU ${gpu_id}，数据 ${FOLD_DIR}，日志根目录 ${LOG_DIR}"
  (
    set -euo pipefail
    python "${ROOT_DIR}/train_elementary.py" \
      --data_path "${FOLD_DIR}" \
      --base_log "${LOG_DIR}" \
      --device "${gpu_id}" \
      "${EXTRA_ARGS[@]}" \
      > "${LOG_FILE}" 2>&1
  ) &
  pid=$!
  PIDS+=("${pid}")
  PIDS_INFO+=("fold_${i}:gpu_${gpu_id}:pid_${pid}:log_${LOG_FILE}")
done

echo "========================================="
echo "[INFO] 等待全部 fold 训练结束..."
fail=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  info="${PIDS_INFO[$idx]}"
  if wait "${pid}"; then
    echo "[INFO] 完成 ${info}"
  else
    echo "[ERROR] 失败 ${info}" >&2
    fail=1
  fi
done

if (( fail != 0 )); then
  echo "[ERROR] 部分 fold 训练失败，请查看对应 train.log" >&2
  exit 1
fi

echo "[INFO] 全部 fold 训练完成"
