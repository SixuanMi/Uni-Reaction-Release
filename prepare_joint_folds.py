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


def unique_by_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"当前数据缺少标识列: {id_col}")
    dup_mask = df.duplicated(subset=[id_col], keep='first')
    dup_cnt = int(dup_mask.sum())
    if dup_cnt > 0:
        print(f"[WARN] 检测到 {dup_cnt} 条重复 {id_col}，仅保留首次出现记录")
    return df.loc[~dup_mask].reset_index(drop=True)


def load_prev_round_split(prev_round_dir: str):
    train_p = os.path.join(prev_round_dir, "train.csv")
    val_p = os.path.join(prev_round_dir, "val.csv")
    test_p = os.path.join(prev_round_dir, "test.csv")
    if not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
        # 兼容传入 folds 根目录的场景，默认使用 fold_1（其 train+val 覆盖上一轮训练池）
        fold1_dir = os.path.join(prev_round_dir, "fold_1")
        train_p = os.path.join(fold1_dir, "train.csv")
        val_p = os.path.join(fold1_dir, "val.csv")
        test_p = os.path.join(fold1_dir, "test.csv")
    if not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
        raise FileNotFoundError(
            f"--prev_round_dir={prev_round_dir} 需直接包含 train/val/test.csv，"
            "或包含 fold_1/train.csv, fold_1/val.csv, fold_1/test.csv"
        )
    prev_train = pd.read_csv(train_p)
    prev_val = pd.read_csv(val_p)
    prev_test = pd.read_csv(test_p)
    return prev_train, prev_val, prev_test


def resolve_dynamic_test(
    df: pd.DataFrame,
    id_col: str,
    test_size: float,
    seed: int,
    prev_round_dir: str
):
    prev_train, prev_val, prev_test = load_prev_round_split(prev_round_dir)
    for name, prev_df in [("train.csv", prev_train), ("val.csv", prev_val), ("test.csv", prev_test)]:
        if id_col not in prev_df.columns:
            raise ValueError(f"{prev_round_dir}/{name} 缺少标识列 {id_col}")

    current_ids = set(df[id_col].astype(str).tolist())
    prev_train_ids = set(prev_train[id_col].astype(str).tolist())
    prev_val_ids = set(prev_val[id_col].astype(str).tolist())
    prev_test_list = prev_test[id_col].astype(str).tolist()  # 保留顺序
    prev_test_ids = set(prev_test_list)
    prev_seen_ids = prev_train_ids | prev_val_ids | prev_test_ids

    prev_test_kept = [x for x in prev_test_list if x in current_ids]
    new_ids_set = current_ids - prev_seen_ids

    total_n = len(df)
    if total_n < 2:
        raise ValueError("样本数少于2，无法切分 train/val/test")
    target_test_n = int(round(total_n * test_size))
    target_test_n = max(1, min(target_test_n, total_n - 1))

    need_extra = target_test_n - len(prev_test_kept)
    extra_test_ids = []
    if need_extra > 0:
        new_candidates = [x for x in df[id_col].astype(str).tolist() if x in new_ids_set]
        if len(new_candidates) <= need_extra:
            extra_test_ids = new_candidates
            print(
                f"[WARN] 新增样本不足以补满目标测试集：需要 {need_extra}，可用 {len(new_candidates)}，"
                "将使用全部新增样本加入 test"
            )
        else:
            rng = np.random.RandomState(seed)
            pick_idx = rng.choice(len(new_candidates), size=need_extra, replace=False)
            pick_idx = set(int(i) for i in pick_idx.tolist())
            extra_test_ids = [x for i, x in enumerate(new_candidates) if i in pick_idx]
    elif need_extra < 0:
        print(
            f"[WARN] 上一轮测试集在当前数据中保留数 {len(prev_test_kept)} 已超过目标 {target_test_n}，"
            "为保持跨轮连续性，本轮不裁剪测试集"
        )

    test_ids_ordered = prev_test_kept + extra_test_ids
    test_ids_set = set(test_ids_ordered)

    # 核心约束：上一轮 train/val 不允许进入本轮 test
    leak_ids = test_ids_set & (prev_train_ids | prev_val_ids)
    if leak_ids:
        raise ValueError(f"检测到泄漏：有 {len(leak_ids)} 个上一轮 train/val 样本进入本轮 test")

    test_df = df[df[id_col].astype(str).isin(test_ids_set)].copy().reset_index(drop=True)
    train_val_df = df[~df[id_col].astype(str).isin(test_ids_set)].copy().reset_index(drop=True)

    print(
        f"[INFO] 动态测试集构建: 目标={target_test_n}, "
        f"继承上一轮test={len(prev_test_kept)}, 新增补充={len(extra_test_ids)}, "
        f"最终test={len(test_df)}, train+val池={len(train_val_df)}"
    )
    print(
        f"[INFO] 历史集合统计: prev_train={len(prev_train_ids)}, "
        f"prev_val={len(prev_val_ids)}, prev_test={len(prev_test_ids)}, "
        f"current_new={len(new_ids_set)}"
    )
    return train_val_df, test_df


def prepare_splits(
    data_path: str, out_root: str, n_folds: int,
    test_size: float, seed: int, prev_round_dir: str = None, id_col: str = "Name"
):
    df = load_all(data_path)
    required_cols = {'Reaction', 'Is_elementary', 'Barrier'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {missing}")

    base_cols = ['Reaction', 'Is_elementary', 'Barrier']
    keep_cols = base_cols
    if prev_round_dir:
        if id_col not in df.columns:
            raise ValueError(f"使用 --prev_round_dir 时，当前数据必须包含标识列 {id_col}")
        # 避免 id_col 与基础列重名（如 id_col=Reaction）导致重复列
        keep_cols = [id_col] + [c for c in base_cols if c != id_col]
    df = df[keep_cols].copy()
    df['Is_elementary'] = df['Is_elementary'].astype(int)

    if prev_round_dir:
        df = unique_by_id(df, id_col=id_col)
        train_val_df, test_df = resolve_dynamic_test(
            df=df,
            id_col=id_col,
            test_size=test_size,
            seed=seed,
            prev_round_dir=prev_round_dir
        )
    else:
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
    parser = argparse.ArgumentParser("准备K折分层数据（支持跨轮动态测试集）")
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
    parser.add_argument(
        '--prev_round_dir', type=str, default=None,
        help='上一轮划分目录（含 train.csv/val.csv/test.csv）；为空时使用原始随机切分逻辑'
    )
    parser.add_argument(
        '--id_col', type=str, default='Name',
        help='用于跨轮去重与防泄漏的样本标识列，仅在 --prev_round_dir 提供时生效'
    )
    args = parser.parse_args(argv)

    prepare_splits(
        data_path=args.data_path,
        out_root=args.output_dir,
        n_folds=args.n_folds,
        test_size=args.test_size,
        seed=args.seed,
        prev_round_dir=args.prev_round_dir,
        id_col=args.id_col
    )


if __name__ == '__main__':
    main()
