import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_all(data_path: str) -> pd.DataFrame:
    combo = os.path.join(data_path, 'combined.csv')
    if os.path.exists(combo):
        df = pd.read_csv(combo)
        print(f"[INFO] 使用 combined.csv, 样本数 {len(df)}")
        return df

    dfs = []
    for part in ['train', 'val', 'test']:
        p = os.path.join(data_path, f'{part}.csv')
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError(
            f"未找到 train/val/test 或 combined.csv 于 {data_path}"
        )
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 合并 train/val/test, 样本数 {len(df)}")
    return df


def prepare_splits(
    data_path: str, out_root: str, n_folds: int,
    test_size: float, seed: int
):
    df = load_all(data_path)
    required_cols = {'Reaction', 'Is_elementary', 'Barrier'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {missing}")

    df = df[['Reaction', 'Is_elementary', 'Barrier']].copy()
    df['Is_elementary'] = df['Is_elementary'].astype(int)
    labels = df['Is_elementary'].values
    indices = np.arange(len(df))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed,
        stratify=labels, shuffle=True
    )
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"[INFO] 全局测试集大小 {len(test_df)}, 训练验证池 {len(train_val_df)}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    os.makedirs(out_root, exist_ok=True)
    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(train_val_df, train_val_df['Is_elementary']),
        start=1
    ):
        fold_dir = os.path.join(out_root, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        train_df = train_val_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)
        print(
            f"[INFO] Fold {fold}: train {len(train_df)}, "
            f"val {len(val_df)}, test {len(test_df)} -> {fold_dir}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser("准备K折分层数据（固定测试集）")
    parser.add_argument(
        '--data_path', required=True,
        help='原始数据目录（含 train/val/test 或 combined.csv）'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='输出根目录（将生成 fold_*/train.csv 等）'
    )
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args(argv)

    prepare_splits(
        data_path=args.data_path,
        out_root=args.output_dir,
        n_folds=args.n_folds,
        test_size=args.test_size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
