#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# 可选环境变量：
#   GPU_IDS=0,1,2,3,4,5,6,7  -> 等价于追加 --devices 0,1,2,3,4,5,6,7
GPU_IDS=${GPU_IDS:-}

HAS_DEVICES_ARG=0
for arg in "$@"; do
  if [[ "${arg}" == "--devices" ]]; then
    HAS_DEVICES_ARG=1
    break
  fi
done

ARGS=("$@")
if [[ -n "${GPU_IDS}" && ${HAS_DEVICES_ARG} -eq 0 ]]; then
  ARGS+=(--devices "${GPU_IDS}")
fi

python "${ROOT_DIR}/vote_infer_unlabeled.py" "${ARGS[@]}"
