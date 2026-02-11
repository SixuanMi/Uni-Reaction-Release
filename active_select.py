import argparse
import os
import pickle

import numpy as np
import pandas as pd


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def sort_pkl_by_uncertainty(pkl_path: str) -> list:
    with open(pkl_path, "rb") as fin:
        data = pickle.load(fin)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("PKL 文件需为包含 reaction_prediction 的列表")

    def metrics(item: dict):
        preds = item.get('reaction_prediction') or []
        if len(preds) == 0:
            raise ValueError("PKL 中存在缺少 reaction_prediction 的样本")
        latest = preds[-1]
        if ('uncert_cls_entropy_rounded' not in latest) or ('uncert_reg_std' not in latest):
            raise ValueError("reaction_prediction 中缺少不确定性指标")
        return float(latest['uncert_cls_entropy_rounded']), float(latest['uncert_reg_std'])

    return sorted(data, key=metrics, reverse=True)


def main():
    parser = argparse.ArgumentParser("主动学习样本选择（分类不确定性优先）")
    parser.add_argument('--input', required=True, help='vote_infer_unlabeled 生成的 CSV（或经过处理的 PKL）')
    parser.add_argument('--top_n', type=int, default=100, help='选择前 N 个不确定样本（PKL 输入时忽略）')
    parser.add_argument('--output', required=True, help='输出 CSV（或 PKL，取决于输入类型）')
    args = parser.parse_args()
    print(args)

    if args.input.endswith(".pkl"):
        sorted_pkl = sort_pkl_by_uncertainty(args.input)
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "wb") as fout:
            pickle.dump(sorted_pkl, fout)
        print(f"[INFO] PKL 输入已按不确定性排序（未截断 Top N），保存至 {args.output}")
        return

    df = pd.read_csv(args.input)
    cls_cols = [c for c in df.columns if c.endswith('_cls_prob')]
    reg_cols = [c for c in df.columns if c.endswith('_barrier')]
    if not cls_cols or not reg_cols:
        raise ValueError("输入需包含 model*_cls_prob 和 model*_barrier 列")

    cls_probs = df[cls_cols].to_numpy(dtype=float)  # [n_samples, n_models]
    reg_preds = df[reg_cols].to_numpy(dtype=float)  # [n_samples, n_models]

    # 分类不确定性：熵（平均概率）
    mean_cls_prob = np.nanmean(cls_probs, axis=1)
    cls_entropy = entropy(mean_cls_prob)
    cls_entropy_rounded = np.round(cls_entropy, 3)
    # cls_entropy_norm = cls_entropy / np.log(2.0)  # 二分类最大熵 log(2)

    # 回归标准差（用于优先排序，避免回归已确定的样本）
    reg_std = np.nanstd(reg_preds, axis=1)
    reg_std_rounded = np.round(reg_std, 2)

    df['uncert_cls_entropy'] = cls_entropy
    df['uncert_cls_entropy_rounded'] = cls_entropy_rounded
    # df['uncert_cls_entropy_norm'] = cls_entropy_norm
    df['uncert_reg_std'] = reg_std_rounded

    # 先按回归标准差降序排序，再取分类熵 Top N
    # 先取分类熵，再按回归标准差降序排序 Top N
    df_sorted = df.sort_values(
        by=['uncert_cls_entropy_rounded', 'uncert_reg_std'],
        ascending=[False, False]
    ).reset_index(drop=True)
    top_df = df_sorted.head(args.top_n)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    top_df.to_csv(args.output, index=False)

    print(f"[INFO] 选取 Top-{args.top_n} 不确定样本，保存至 {args.output}")
    print(f"[INFO] 回归 std 统计: mean={np.nanmean(reg_std):.4f}, max={np.nanmax(reg_std):.4f}, min={np.nanmin(reg_std):.4f}")
    print(f"[INFO] 分类熵 统计: mean={np.nanmean(cls_entropy):.4f}, max={np.nanmax(cls_entropy):.4f}, min={np.nanmin(cls_entropy):.4f}")
    print(f"[INFO] Top-{args.top_n} 分类熵: mean={top_df['uncert_cls_entropy'].mean():.4f}, max={top_df['uncert_cls_entropy'].max():.4f}")


if __name__ == '__main__':
    main()
