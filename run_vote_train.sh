#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_vote_train.sh DATA_PATH [EXTRA_TRAIN_ARGS...]
# DATA_PATH: 原始数据目录（包含 train/val/test 或 combined.csv）
# 可通过环境变量控制: N_FOLDS(默认5), TEST_SIZE(默认0.1), SEED(默认2025)

DATA_PATH=${1:?please provide data path}
shift || true

N_FOLDS=${N_FOLDS:-5}
TEST_SIZE=${TEST_SIZE:-0.1}
SEED=${SEED:-2025}

TS=$(date +%s)
SPLIT_DIR="fold_splits_${TS}"

echo "[INFO] 生成分层拆分: 全局测试占比 ${TEST_SIZE}, 折数 ${N_FOLDS}, 输出 ${SPLIT_DIR}"
python - <<'PY' "$DATA_PATH" "$SPLIT_DIR" "$N_FOLDS" "$TEST_SIZE" "$SEED"
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

data_path, out_root, n_folds, test_size, seed = sys.argv[1:]
test_size = float(test_size)
n_folds = int(n_folds)
seed = int(seed)

def load_all(path):
    combo = os.path.join(path, 'combined.csv')
    if os.path.exists(combo):
        df = pd.read_csv(combo)
        print(f"[INFO] 使用 combined.csv, 样本数 {len(df)}")
        return df
    dfs = []
    for part in ['train', 'val', 'test']:
        p = os.path.join(path, f'{part}.csv')
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError(f"未找到 train/val/test 或 combined.csv 于 {path}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 合并 train/val/test, 样本数 {len(df)}")
    return df

df = load_all(data_path)
required_cols = {'Reaction', 'Is_elementary', 'Barrier'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"缺少列: {missing}")

# 仅保留必要列，确保标签为 int
df = df[['Reaction', 'Is_elementary', 'Barrier']].copy()
df['Is_elementary'] = df['Is_elementary'].astype(int)
labels = df['Is_elementary'].values
indices = np.arange(len(df))

train_val_idx, test_idx = train_test_split(
    indices, test_size=test_size, random_state=seed, stratify=labels, shuffle=True
)
train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

print(f"[INFO] 全局测试集大小 {len(test_df)}, 训练验证池 {len(train_val_df)}")

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
os.makedirs(out_root, exist_ok=True)
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['Is_elementary']), start=1):
    fold_dir = os.path.join(out_root, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)
    train_df = train_val_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
    print(f"[INFO] Fold {fold}: train {len(train_df)}, val {len(val_df)}, test {len(test_df)} -> {fold_dir}")
PY

# 依次训练每个折的模型，日志/模型目录避免覆盖
EXTRA_ARGS=("$@")
for i in $(seq 1 "$N_FOLDS"); do
  FOLD_DIR="${SPLIT_DIR}/fold_${i}"
  LOG_DIR="log_joint_fold${i}_${TS}"
  echo "[INFO] 训练折 ${i}/${N_FOLDS}，数据 ${FOLD_DIR}，日志根目录 ${LOG_DIR}"
  python train_elementary.py --data_path "${FOLD_DIR}" --base_log "${LOG_DIR}" "${EXTRA_ARGS[@]}"
done
