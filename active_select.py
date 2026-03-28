import argparse
import os
import re

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


def parse_line_id_from_origin_idx(series: pd.Series) -> pd.Series:
    prefix = series.astype(str).str.split('_', n=1).str[0]
    nums = prefix.apply(
        lambda x: int(m.group(1)) if (m := re.search(r'(\d+)', x)) else np.nan
    )
    return pd.to_numeric(nums, errors='coerce')


def main():
    parser = argparse.ArgumentParser("主动学习样本选择（分类不确定性优先）")
    parser.add_argument('--input', required=True, help='vote_infer_unlabeled 生成的 CSV')
    parser.add_argument('--top_n', type=int, default=100, help='选择前 N 个不确定样本')
    parser.add_argument('--output', required=True, help='输出 CSV')
    parser.add_argument(
        '--uncertainty_metric',
        type=str,
        default='bald',
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
    parser.add_argument(
        '--cls_threshold',
        type=float,
        default=None,
        help='分类不确定性阈值，仅保留 cls_metric >= cls_threshold 的样本；最终选择为 TopN 与阈值集合的交集'
    )
    parser.add_argument(
        '--line_id_col',
        type=str,
        default='origin_idx',
        help='用于提取 line 编号的列名（默认 origin_idx，格式如 line116_xxx）'
    )
    parser.add_argument(
        '--line_min',
        type=int,
        default=1,
        help='line 编号最小值（用于输出阈值以上 line 列表）'
    )
    parser.add_argument(
        '--line_max',
        type=int,
        default=168,
        help='line 编号最大值（用于输出阈值以上 line 列表）'
    )
    parser.add_argument(
        '--line_metric_stats_output',
        type=str,
        default=None,
        help='line 编号与分类指标关系统计 CSV 输出路径（默认: <output_basename>_line_metric_stats.csv）'
    )
    parser.add_argument(
        '--line_ids_output',
        type=str,
        default=None,
        help='阈值以上 line 编号列表 TXT 输出路径（默认: <output_basename>_line_ids_ge_threshold.txt）'
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

    df['_row_id'] = np.arange(len(df), dtype=int)

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

    # line 编号提取
    if args.line_id_col in df.columns:
        df['line_id'] = parse_line_id_from_origin_idx(df[args.line_id_col])
    else:
        df['line_id'] = np.nan
        print(f"[WARN] 未找到 line_id_col={args.line_id_col}，将跳过 line 编号统计")

    # 排序使用原始值，round 列只用于展示
    df_sorted = df.sort_values(
        by=[metric_col, reg_metric_col],
        ascending=[False, False]
    ).reset_index(drop=True)
    top_df = df_sorted.head(args.top_n).copy()

    threshold = args.cls_threshold
    if threshold is not None:
        top_df = top_df[top_df[metric_col] >= threshold].reset_index(drop=True)
        threshold_pool_mask = (df[metric_col] >= threshold).to_numpy(dtype=bool)
    else:
        threshold_pool_mask = np.ones(len(df), dtype=bool)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    top_df.drop(columns=['_row_id'], errors='ignore').to_csv(args.output, index=False)

    # line 编号与分类指标关系统计
    output_base, _ = os.path.splitext(args.output)
    line_metric_stats_output = (
        args.line_metric_stats_output
        if args.line_metric_stats_output
        else f"{output_base}_line_metric_stats.csv"
    )
    line_ids_output = (
        args.line_ids_output
        if args.line_ids_output
        else f"{output_base}_line_ids_ge_threshold.txt"
    )

    line_stats_df = df[['line_id', metric_col, '_row_id']].copy().dropna(subset=['line_id'])
    if not line_stats_df.empty:
        line_stats_df['line_id'] = line_stats_df['line_id'].astype(int)
        line_stats_df['above_threshold'] = threshold_pool_mask[line_stats_df['_row_id'].to_numpy(dtype=int)]
        selected_row_ids = set(top_df['_row_id'].tolist())
        line_stats_df['in_final_selection'] = line_stats_df['_row_id'].isin(selected_row_ids)
        line_metric_stats = line_stats_df.groupby('line_id', observed=True).agg(
            sample_count=(metric_col, 'size'),
            cls_metric_mean=(metric_col, 'mean'),
            cls_metric_median=(metric_col, 'median'),
            cls_metric_p90=(metric_col, lambda x: np.nanpercentile(x, 90)),
            cls_metric_max=(metric_col, 'max'),
            above_threshold_count=('above_threshold', 'sum'),
            in_final_selection_count=('in_final_selection', 'sum')
        ).reset_index()
        line_metric_stats['above_threshold_ratio'] = (
            line_metric_stats['above_threshold_count'] / line_metric_stats['sample_count']
        )
        line_metric_stats['in_final_selection_ratio'] = (
            line_metric_stats['in_final_selection_count'] / line_metric_stats['sample_count']
        )
        line_metric_stats = line_metric_stats.sort_values(
            by=['cls_metric_mean', 'sample_count'], ascending=[False, False]
        )
        line_metric_stats.to_csv(line_metric_stats_output, index=False)
    else:
        pd.DataFrame(columns=[
            'line_id', 'sample_count', 'cls_metric_mean', 'cls_metric_median',
            'cls_metric_p90', 'cls_metric_max', 'above_threshold_count',
            'in_final_selection_count', 'above_threshold_ratio', 'in_final_selection_ratio'
        ]).to_csv(line_metric_stats_output, index=False)

    # 输出“阈值以上且在 [line_min, line_max]”出现过的 line 编号，一行一个数字
    line_pool_df = df.loc[threshold_pool_mask, ['line_id']].dropna()
    if not line_pool_df.empty:
        line_pool = line_pool_df['line_id'].astype(int)
        line_pool = sorted(set(line_pool[(line_pool >= args.line_min) & (line_pool <= args.line_max)]))
    else:
        line_pool = []
    with open(line_ids_output, 'w', encoding='utf-8') as f:
        for line_id in line_pool:
            f.write(f"{line_id}\n")

    print(
        f"[INFO] 选取样本 = Top-{args.top_n} 与 "
        f"{metric_name}>={threshold if threshold is not None else '-inf'} 的交集，"
        f"保存至 {args.output}"
    )
    print(f"[INFO] 最终选中样本数: {len(top_df)}")
    print(f"[INFO] line 关系统计已保存: {line_metric_stats_output}")
    print(f"[INFO] 阈值以上 line({args.line_min}-{args.line_max}) 列表已保存: {line_ids_output}")
    if threshold is not None:
        print(f"[INFO] 全量候选中 {metric_name}>={threshold} 的样本数: {int(np.sum(threshold_pool_mask))}")
        print(f"[INFO] 其中 line({args.line_min}-{args.line_max}) 覆盖数: {len(line_pool)}")
    print(f"[INFO] 回归 std 统计: mean={np.nanmean(reg_std):.4f}, max={np.nanmax(reg_std):.4f}, min={np.nanmin(reg_std):.4f}")
    print(
        f"[INFO] 回归 std 分桶z统计: mean={np.nanmean(reg_std_binned_z):.4f}, "
        f"max={np.nanmax(reg_std_binned_z):.4f}, min={np.nanmin(reg_std_binned_z):.4f}"
    )
    print(f"[INFO] 分类熵 统计: mean={np.nanmean(cls_entropy):.4f}, max={np.nanmax(cls_entropy):.4f}, min={np.nanmin(cls_entropy):.4f}")
    print(f"[INFO] 期望熵 统计: mean={np.nanmean(cls_expected_entropy):.4f}, max={np.nanmax(cls_expected_entropy):.4f}, min={np.nanmin(cls_expected_entropy):.4f}")
    print(f"[INFO] BALD 统计: mean={np.nanmean(cls_bald):.4f}, max={np.nanmax(cls_bald):.4f}, min={np.nanmin(cls_bald):.4f}")
    if len(top_df) > 0:
        print(f"[INFO] Top交集 {metric_name}: mean={top_df[metric_col].mean():.4f}, max={top_df[metric_col].max():.4f}")
        print(f"[INFO] Top交集 {reg_metric_name}: mean={top_df[reg_metric_col].mean():.4f}, max={top_df[reg_metric_col].max():.4f}")
    else:
        print(f"[WARN] TopN与阈值交集为空：请降低 --cls_threshold 或提高 --top_n")


if __name__ == '__main__':
    main()
