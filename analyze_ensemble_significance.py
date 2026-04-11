import argparse
import json
from typing import Dict

import numpy as np

try:
    from scipy.stats import ttest_ind
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_metric_vectors(data: Dict, source_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    if "per_model_metric_vectors" in data:
        metric_vectors = data["per_model_metric_vectors"]
        out = {"classification": {}, "regression": {}}
        for group in out.keys():
            group_data = metric_vectors.get(group, {})
            for metric_name, values in group_data.items():
                out[group][metric_name] = np.asarray(values, dtype=float)
        return out

    if "per_model" in data:
        out = {"classification": {}, "regression": {}}
        for model_item in data["per_model"]:
            cls_metrics = model_item.get("classification", {}).get("metrics", {})
            reg_metrics = model_item.get("regression", {}).get("metrics", {})
            for k, v in cls_metrics.items():
                if isinstance(v, (int, float)):
                    out["classification"].setdefault(k, []).append(float(v))
            for k, v in reg_metrics.items():
                if isinstance(v, (int, float)):
                    out["regression"].setdefault(k, []).append(float(v))
        for group in out.keys():
            for metric_name in list(out[group].keys()):
                out[group][metric_name] = np.asarray(out[group][metric_name], dtype=float)
        return out

    raise ValueError(
        f"{source_path} 中未找到 per_model_metric_vectors/per_model，"
        "请先使用更新后的 predict_ensemble.py 重新导出结果"
    )


def independent_ttest(a: np.ndarray, b: np.ndarray, alternative: str) -> Dict[str, float]:
    a_valid = a[np.isfinite(a)]
    b_valid = b[np.isfinite(b)]
    n_a = int(a_valid.size)
    n_b = int(b_valid.size)
    result = {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": float(np.mean(a_valid)) if n_a > 0 else float("nan"),
        "mean_b": float(np.mean(b_valid)) if n_b > 0 else float("nan"),
        "delta_mean_b_minus_a": (
            float(np.mean(b_valid) - np.mean(a_valid))
            if (n_a > 0 and n_b > 0) else float("nan")
        ),
        "p_ttest_ind": float("nan")
    }
    if n_a >= 2 and n_b >= 2:
        try:
            # 语义: 以 B 相对 A 的方向进行检验
            result["p_ttest_ind"] = float(
                ttest_ind(
                    b_valid, a_valid,
                    equal_var=True,
                    nan_policy="omit",
                    alternative=alternative
                ).pvalue
            )
        except TypeError:
            # 兼容旧版 SciPy（不支持 alternative 参数）
            out = ttest_ind(b_valid, a_valid, equal_var=True, nan_policy="omit")
            p_two = float(out.pvalue)
            t_stat = float(out.statistic)
            if alternative == "two-sided":
                result["p_ttest_ind"] = p_two
            elif alternative == "greater":
                result["p_ttest_ind"] = p_two / 2.0 if t_stat >= 0 else 1.0 - p_two / 2.0
            elif alternative == "less":
                result["p_ttest_ind"] = p_two / 2.0 if t_stat <= 0 else 1.0 - p_two / 2.0
        except Exception:
            pass
    return result


def fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.6g}"


def main():
    parser = argparse.ArgumentParser("比较两轮 ensemble 结果中 8 个子模型指标的显著性（独立样本 t 检验）")
    parser.add_argument("--a", required=True, help="基线 JSON（如上一轮 ensemble_result.json）")
    parser.add_argument("--b", required=True, help="对比 JSON（如下一轮 ensemble_result.json）")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性阈值，默认 0.05")
    parser.add_argument(
        "--alternative",
        type=str,
        default="greater",
        choices=["greater", "less", "two-sided"],
        help="t 检验方向（相对 A 看 B）：greater=检验 B>A（默认），less=检验 B<A，two-sided=双端检验"
    )
    parser.add_argument("--output", type=str, default=None, help="可选：将完整检验结果写入 JSON")
    args = parser.parse_args()

    if not HAS_SCIPY:
        raise ImportError("当前环境缺少 scipy，无法进行 t 检验。请先安装 scipy。")

    data_a = load_json(args.a)
    data_b = load_json(args.b)
    metrics_a = extract_metric_vectors(data_a, args.a)
    metrics_b = extract_metric_vectors(data_b, args.b)

    report = {
        "input_a": args.a,
        "input_b": args.b,
        "alpha": args.alpha,
        "alternative": args.alternative,
        "results": {}
    }

    print("=" * 110)
    print("[Significance Report] Pairwise comparison of per-model metrics (B vs A)")
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"alpha: {args.alpha}")
    print(f"alternative: {args.alternative} (direction is B relative to A)")
    print("tests: independent two-sample t-test (equal_var=True)")
    print("=" * 110)

    for group in ["classification", "regression"]:
        common_metrics = sorted(set(metrics_a.get(group, {}).keys()) & set(metrics_b.get(group, {}).keys()))
        report["results"][group] = {}

        print(f"\n[{group}]")
        if not common_metrics:
            print("  no common scalar metrics found")
            continue

        for metric_name in common_metrics:
            arr_a = metrics_a[group][metric_name]
            arr_b = metrics_b[group][metric_name]

            stats = independent_ttest(arr_a, arr_b, alternative=args.alternative)
            report["results"][group][metric_name] = stats

            p_main = stats["p_ttest_ind"]
            significant = np.isfinite(p_main) and (p_main < args.alpha)
            flag = "*" if significant else "-"
            print(
                f"  {flag} {metric_name:20s} "
                f"nA={stats['n_a']:2d} "
                f"nB={stats['n_b']:2d} "
                f"meanA={fmt(stats['mean_a'])} "
                f"meanB={fmt(stats['mean_b'])} "
                f"delta={fmt(stats['delta_mean_b_minus_a'])} "
                f"p_t={fmt(stats['p_ttest_ind'])}"
            )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] 已写出详细结果: {args.output}")


if __name__ == "__main__":
    main()
