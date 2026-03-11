import argparse
import os

import numpy as np
import pandas as pd


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def classification_uncertainties(cls_probs: np.ndarray):
    """返回分类不确定性：预测熵、期望熵、BALD（互信息近似）。"""
    mean_cls_prob = np.nanmean(cls_probs, axis=1)
    pred_entropy = entropy(mean_cls_prob)
    expected_entropy = np.nanmean(entropy(cls_probs), axis=1)
    bald = np.maximum(pred_entropy - expected_entropy, 0.0)
    return pred_entropy, expected_entropy, bald


def regression_std_binned_zscore(reg_mean: np.ndarray, reg_std: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    按 |reg_mean| 分桶，在每个桶内对 reg_std 做 robust z-score:
      z = (std - median_bin) / (IQR_bin + eps)
    用于降低“均值越大、std 天然越大”带来的排序偏置。
    """
    out = np.full_like(reg_std, fill_value=np.nan, dtype=float)
    valid = np.isfinite(reg_mean) & np.isfinite(reg_std)
    if not np.any(valid):
        return np.zeros_like(reg_std, dtype=float)

    abs_mean = np.abs(reg_mean[valid])
    valid_std = reg_std[valid]
    q = min(n_bins, int(valid.sum()))

    # qcut 自动做分位分桶，重复边界会自动降桶数
    try:
        bin_codes = pd.qcut(abs_mean, q=q, labels=False, duplicates='drop')
        bin_codes = np.asarray(bin_codes, dtype=int)
    except ValueError:
        bin_codes = np.zeros(valid_std.shape[0], dtype=int)

    global_iqr = np.nanpercentile(valid_std, 75) - np.nanpercentile(valid_std, 25)
    if not np.isfinite(global_iqr) or global_iqr < 1e-8:
        global_iqr = np.nanstd(valid_std)
    if not np.isfinite(global_iqr) or global_iqr < 1e-8:
        global_iqr = 1.0

    z = np.zeros_like(valid_std, dtype=float)
    for b in np.unique(bin_codes):
        mask = (bin_codes == b)
        bucket = valid_std[mask]
        med = np.nanmedian(bucket)
        iqr = np.nanpercentile(bucket, 75) - np.nanpercentile(bucket, 25)
        if not np.isfinite(iqr) or iqr < 1e-8:
            iqr = global_iqr
        z[mask] = (bucket - med) / iqr

    out[valid] = z
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser("主动学习样本选择（分类不确定性优先）")
    parser.add_argument('--input', required=True, help='vote_infer_unlabeled 生成的 CSV')
    parser.add_argument('--top_n', type=int, default=100, help='选择前 N 个不确定样本')
    parser.add_argument('--output', required=True, help='输出 CSV')
    parser.add_argument(
        '--uncertainty_metric',
        type=str,
        default='entropy',
        choices=['entropy', 'bald'],
        help='分类不确定性指标：entropy=预测熵，bald=BALD互信息近似'
    )
    parser.add_argument(
        '--reg_uncertainty_metric',
        type=str,
        default='binned_z',
        choices=['std', 'binned_z'],
        help='回归次级排序指标：std=原始标准差，binned_z=按|均值|分10桶后的桶内标准化'
    )
    args = parser.parse_args()
    print(args)

    if not args.input.endswith(".csv"):
        raise ValueError("active_select 仅支持 CSV 输入，请传入 .csv 文件")
    if not args.output.endswith(".csv"):
        raise ValueError("active_select 仅支持 CSV 输出，请将 --output 设为 .csv 文件")

    df = pd.read_csv(args.input)
    cls_cols = [c for c in df.columns if c.endswith('_cls_prob')]
    reg_cols = [c for c in df.columns if c.endswith('_barrier')]
    if not cls_cols or not reg_cols:
        raise ValueError("输入需包含 model*_cls_prob 和 model*_barrier 列")

    cls_probs = df[cls_cols].to_numpy(dtype=float)  # [n_samples, n_models]
    reg_preds = df[reg_cols].to_numpy(dtype=float)  # [n_samples, n_models]

    # 分类不确定性
    cls_entropy, cls_expected_entropy, cls_bald = classification_uncertainties(cls_probs)
    cls_entropy_rounded = np.round(cls_entropy, 3)
    cls_expected_entropy_rounded = np.round(cls_expected_entropy, 3)
    cls_bald_rounded = np.round(cls_bald, 3)

    # 回归标准差（作为次级排序键）
    reg_mean = np.nanmean(reg_preds, axis=1)
    reg_std = np.nanstd(reg_preds, axis=1)
    reg_std_binned_z = regression_std_binned_zscore(reg_mean, reg_std, n_bins=10)
    reg_std_rounded = np.round(reg_std, 2)
    reg_std_binned_z_rounded = np.round(reg_std_binned_z, 3)

    df['uncert_reg_mean'] = reg_mean
    df['uncert_cls_entropy'] = cls_entropy
    df['uncert_cls_entropy_rounded'] = cls_entropy_rounded
    df['uncert_cls_expected_entropy'] = cls_expected_entropy
    df['uncert_cls_expected_entropy_rounded'] = cls_expected_entropy_rounded
    df['uncert_cls_bald'] = cls_bald
    df['uncert_cls_bald_rounded'] = cls_bald_rounded
    df['uncert_reg_std'] = reg_std
    df['uncert_reg_std_rounded'] = reg_std_rounded
    df['uncert_reg_std_binned_z'] = reg_std_binned_z
    df['uncert_reg_std_binned_z_rounded'] = reg_std_binned_z_rounded

    metric_col = 'uncert_cls_entropy' if args.uncertainty_metric == 'entropy' else 'uncert_cls_bald'
    metric_name = '分类熵' if args.uncertainty_metric == 'entropy' else 'BALD'
    reg_metric_col = 'uncert_reg_std' if args.reg_uncertainty_metric == 'std' else 'uncert_reg_std_binned_z'
    reg_metric_name = '回归std' if args.reg_uncertainty_metric == 'std' else '回归分桶标准化std(z)'
    if args.uncertainty_metric == 'bald' and len(cls_cols) < 2:
        print("[WARN] 仅检测到 1 个分类模型，BALD 退化为 0，建议至少使用 2 个模型")

    # 排序使用原始值，round 列只用于展示
    df_sorted = df.sort_values(
        by=[metric_col, reg_metric_col],
        ascending=[False, False]
    ).reset_index(drop=True)
    top_df = df_sorted.head(args.top_n)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    top_df.to_csv(args.output, index=False)

    print(
        f"[INFO] 选取 Top-{args.top_n} 不确定样本（分类指标: {args.uncertainty_metric}, "
        f"回归次排序: {args.reg_uncertainty_metric}），保存至 {args.output}"
    )
    print(f"[INFO] 回归 std 统计: mean={np.nanmean(reg_std):.4f}, max={np.nanmax(reg_std):.4f}, min={np.nanmin(reg_std):.4f}")
    print(
        f"[INFO] 回归 std 分桶z统计: mean={np.nanmean(reg_std_binned_z):.4f}, "
        f"max={np.nanmax(reg_std_binned_z):.4f}, min={np.nanmin(reg_std_binned_z):.4f}"
    )
    print(f"[INFO] 分类熵 统计: mean={np.nanmean(cls_entropy):.4f}, max={np.nanmax(cls_entropy):.4f}, min={np.nanmin(cls_entropy):.4f}")
    print(f"[INFO] 期望熵 统计: mean={np.nanmean(cls_expected_entropy):.4f}, max={np.nanmax(cls_expected_entropy):.4f}, min={np.nanmin(cls_expected_entropy):.4f}")
    print(f"[INFO] BALD 统计: mean={np.nanmean(cls_bald):.4f}, max={np.nanmax(cls_bald):.4f}, min={np.nanmin(cls_bald):.4f}")
    print(f"[INFO] Top-{args.top_n} {metric_name}: mean={top_df[metric_col].mean():.4f}, max={top_df[metric_col].max():.4f}")
    print(f"[INFO] Top-{args.top_n} {reg_metric_name}: mean={top_df[reg_metric_col].mean():.4f}, max={top_df[reg_metric_col].max():.4f}")


if __name__ == '__main__':
    main()
