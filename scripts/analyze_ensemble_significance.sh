#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

python "${ROOT_DIR}/analyze_ensemble_significance.py" "$@"

# 用法示例：
# ./scripts/analyze_ensemble_significance.sh \
#   --a /path/to/prev_round/ensemble_result.json \
#   --b /path/to/next_round/ensemble_result.json \
#   --alpha 0.05 \
#   --output /path/to/significance_report.json
