import argparse
import math
import os

import numpy as np
import pandas as pd


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    parser = argparse.ArgumentParser("主动学习样本选择（分类不确定性优先）")
    parser.add_argument('--input', required=True, help='vote_infer_unlabeled 生成的 CSV，包含 model*_cls_prob / model*_barrier')
    parser.add_argument('--top_n', type=int, default=100, help='选择前 N 个不确定样本')
    parser.add_argument('--output', required=True, help='输出 CSV，附加不确定性指标与排序')
    args = parser.parse_args()
    print(args)

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
    cls_entropy_norm = cls_entropy / math.log(2.0)  # 二分类最大熵 log(2)

    # 回归标准差（用于优先排序，避免回归已确定的样本）
    reg_std = np.nanstd(reg_preds, axis=1)

    df['uncert_cls_entropy'] = cls_entropy
    df['uncert_cls_entropy_norm'] = cls_entropy_norm
    df['uncert_reg_std'] = reg_std

    # 先按回归标准差降序排序，再取分类熵 Top N
    df_sorted = df.sort_values(by='uncert_reg_std', ascending=False).reset_index(drop=True)
    df_sorted = df_sorted.sort_values(by='uncert_cls_entropy', ascending=False).reset_index(drop=True)
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
