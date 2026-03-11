import argparse
import glob
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch

from utils.data_utils import load_joint_data_one, fix_seed
from utils.model_factory import build_joint_model, resolve_device
from utils.training.training import eval_joint
from utils.Dataset import joint_colfn
from torch.utils.data import DataLoader

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def collect_model_paths(model_paths: List[str], model_root: str) -> List[str]:
    if model_paths:
        return model_paths
    if model_root:
        paths = glob.glob(os.path.join(model_root, "**", "best_model.pth"), recursive=True)
        if not paths:
            paths = glob.glob(os.path.join(model_root, "**", "best_model.pt"), recursive=True)
        return sorted(paths)
    raise ValueError("未找到模型路径，请提供 --model_paths 或 --main_dir（默认搜索 logs 子目录）")


def parse_device_ids(devices: str, fallback_device: int) -> List[int]:
    if devices is None:
        return [fallback_device]
    ids = []
    for token in devices.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError as e:
            raise ValueError(f"--devices 中存在非法设备编号: {token}") from e
    if not ids:
        raise ValueError("--devices 不能为空，请提供如 0,1,2 的设备列表")
    return ids


def same_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all((a == b) | both_nan))


def infer_one_model(
    model_path: str,
    data_path: str,
    device_id: int,
    model_cfg: Dict,
    bs: int,
    num_worker: int,
    total_heads: int,
    local_heads: int,
    seed: int
) -> Dict:
    fix_seed(seed)
    device = resolve_device(device_id)

    test_set = load_joint_data_one(data_path, 'test')
    test_loader = DataLoader(
        test_set, batch_size=bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=num_worker,
        pin_memory=(device.type == 'cuda')
    )

    model_args = SimpleNamespace(**model_cfg)
    model = build_joint_model(model_args, dropout=0.0).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    res = eval_joint(
        loader=test_loader,
        model=model,
        device=device,
        total_heads=total_heads,
        local_heads=local_heads,
        return_raw=True,
        pos_label=1,
        lambda_reg=0.005
    )
    return {
        'model_path': model_path,
        'device': str(device),
        'cls_pred': res['raw']['cls_pred'],
        'cls_scores': res['raw']['cls_scores'],
        'reg_pred': res['raw']['reg_pred'],
        'cls_true': res['raw']['cls_true'],
        'reg_true': res['raw']['reg_true']
    }


def majority_vote_cls(cls_preds: np.ndarray, tie_break_prob: np.ndarray = None, cls_threshold: float = 0.5) -> np.ndarray:
    # cls_preds: [n_models, n_samples]
    out = []
    for i in range(cls_preds.shape[1]):
        votes = np.bincount(cls_preds[:, i].astype(int), minlength=2)
        if votes[0] == votes[1]:
            if tie_break_prob is not None:
                out.append(int(tie_break_prob[i] >= cls_threshold))
            else:
                out.append(0)
        else:
            out.append(int(np.argmax(votes)))
    return np.array(out, dtype=int)


def soft_vote_cls(cls_scores: np.ndarray, cls_threshold: float = 0.5) -> np.ndarray:
    # cls_scores: [n_models, n_samples], each value is positive-class probability
    mean_prob = np.nanmean(cls_scores, axis=0)
    return (mean_prob >= cls_threshold).astype(int)


def mean_reg(all_reg: np.ndarray) -> np.ndarray:
    # all_reg: [n_models, n_samples]
    if all_reg.size == 0 or all_reg.shape[0] == 0:
        return np.array([])
    valid_mask = np.isfinite(all_reg)
    valid_count = valid_mask.sum(axis=0)
    out = np.full(all_reg.shape[1], np.nan, dtype=np.float64)
    has_valid = valid_count > 0
    if np.any(has_valid):
        valid_values = np.where(valid_mask, all_reg, 0.0)
        out[has_valid] = valid_values[:, has_valid].sum(axis=0) / valid_count[has_valid]
    return out


def std_reg(all_reg: np.ndarray, mean_reg_values: np.ndarray) -> np.ndarray:
    # all_reg: [n_models, n_samples]
    if all_reg.size == 0 or all_reg.shape[0] == 0:
        return np.array([])
    valid_mask = np.isfinite(all_reg)
    valid_count = valid_mask.sum(axis=0)
    out = np.full(all_reg.shape[1], np.nan, dtype=np.float64)
    has_valid = valid_count > 0
    if np.any(has_valid):
        centered = np.where(
            valid_mask[:, has_valid],
            all_reg[:, has_valid] - mean_reg_values[has_valid],
            0.0
        )
        var = (centered ** 2).sum(axis=0) / valid_count[has_valid]
        out[has_valid] = np.sqrt(var)
    return out


def main():
    parser = argparse.ArgumentParser("多模型投票/平均评估")
    parser.add_argument('--main_dir', type=str, default=None, help='训练输出的根目录（如 vote_run_xxx，自动找 folds/test 和 logs）')
    parser.add_argument('--data_path', type=str, default=None, help='自定义测试集目录（含 test.csv），覆盖 main_dir 默认')
    parser.add_argument('--model_paths', nargs='+', default=None, help='直接提供模型路径列表，覆盖 main_dir 默认搜索')
    parser.add_argument('--output', type=str, default=None, help='输出结果 json（默认 main_dir/ensemble_result.json）')
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--devices', type=str, default=None, help='逗号分隔设备ID，如 0,1,2；用于多卡并行推理')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--local_heads', type=int, default=4)
    parser.add_argument(
        '--vote_mode', type=str, default='soft', choices=['hard', 'soft'],
        help='分类集成方式：hard=多数票，soft=平均概率后按阈值判定（推荐）'
    )
    parser.add_argument(
        '--cls_threshold', type=float, default=0.5,
        help='soft 投票阈值（mean_prob >= threshold 判为正类）'
    )
    parser.add_argument(
        '--fusion_mode', type=str, default='legacy', choices=['legacy', 'film'],
        help='R/P融合方式（需与训练一致）'
    )
    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)
    device_ids = parse_device_ids(args.devices, args.device)
    print(f"[INFO] 推理设备: {device_ids}")

    # 解析默认目录
    default_data = None
    default_model_root = None
    if args.main_dir:
        default_data = os.path.join(args.main_dir, "folds", "fold_1")
        default_model_root = os.path.join(args.main_dir, "logs")

    data_path = args.data_path if args.data_path else default_data
    model_root = default_model_root
    output_path = args.output if args.output else (os.path.join(args.main_dir, "ensemble_result.json") if args.main_dir else None)

    paths = collect_model_paths(args.model_paths, model_root)
    if len(paths) == 0:
        raise ValueError("未找到任何模型")
    print(f"[INFO] 模型数: {len(paths)}")

    # 加载测试集
    if not data_path:
        raise ValueError("请提供 --data_path 或 --main_dir")
    test_set = load_joint_data_one(data_path, 'test')
    n_samples = len(test_set)
    print(f"[INFO] 测试样本数: {n_samples}")

    model_cfg = {
        'dim': args.dim,
        'heads': args.heads,
        'n_layer': args.n_layer,
        'negative_slope': args.negative_slope,
        'fusion_mode': args.fusion_mode
    }

    per_model_results = [None] * len(paths)
    if len(device_ids) == 1:
        single_device = device_ids[0]
        print(f"[INFO] 单卡推理模式: device={single_device}")
        for idx, p in enumerate(paths):
            res = infer_one_model(
                model_path=p,
                data_path=data_path,
                device_id=single_device,
                model_cfg=model_cfg,
                bs=args.bs,
                num_worker=args.num_worker,
                total_heads=args.heads,
                local_heads=args.local_heads,
                seed=args.seed
            )
            per_model_results[idx] = res
            print(f"[INFO] 模型推理完成 ({idx + 1}/{len(paths)}): {p} @ {res['device']}")
    else:
        max_parallel = min(len(paths), len(device_ids))
        per_proc_num_worker = args.num_worker // max_parallel if args.num_worker > 0 else 0
        print(
            f"[INFO] 多卡并行推理模式: 并行进程={max_parallel}, "
            f"每进程DataLoader workers={per_proc_num_worker}"
        )
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=max_parallel, mp_context=ctx) as executor:
            future_to_meta = {}
            for idx, p in enumerate(paths):
                assigned_device = device_ids[idx % len(device_ids)]
                fut = executor.submit(
                    infer_one_model,
                    p,
                    data_path,
                    assigned_device,
                    model_cfg,
                    args.bs,
                    per_proc_num_worker,
                    args.heads,
                    args.local_heads,
                    args.seed
                )
                future_to_meta[fut] = (idx, p, assigned_device)

            finished = 0
            for fut in as_completed(future_to_meta):
                idx, p, assigned_device = future_to_meta[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    raise RuntimeError(f"模型推理失败: path={p}, device={assigned_device}") from e
                per_model_results[idx] = res
                finished += 1
                print(
                    f"[INFO] 模型推理完成 ({finished}/{len(paths)}): "
                    f"{p} @ {res['device']}"
                )

    cls_preds = np.stack([np.array(x['cls_pred']) for x in per_model_results], axis=0)
    cls_scores = np.stack([np.array(x['cls_scores']) for x in per_model_results], axis=0)
    reg_preds = np.stack([np.array(x['reg_pred']) for x in per_model_results], axis=0)
    true_cls = np.array(per_model_results[0]['cls_true'])
    true_reg = np.array(per_model_results[0]['reg_true'])
    for i, r in enumerate(per_model_results[1:], start=1):
        cls_true_i = np.array(r['cls_true'])
        reg_true_i = np.array(r['reg_true'])
        if not np.array_equal(true_cls, cls_true_i):
            raise ValueError(f"第 {i + 1} 个模型返回的 cls_true 与第1个模型不一致")
        if not same_with_nan(true_reg, reg_true_i):
            raise ValueError(f"第 {i + 1} 个模型返回的 reg_true 与第1个模型不一致")

    # 集成
    mean_prob_per_sample = np.nanmean(cls_scores, axis=0)  # [n_samples]
    if args.vote_mode == 'soft':
        vote_cls = soft_vote_cls(cls_scores, cls_threshold=args.cls_threshold)
    else:
        vote_cls = majority_vote_cls(
            cls_preds,
            tie_break_prob=mean_prob_per_sample,
            cls_threshold=args.cls_threshold
        )

    mean_reg_pred = mean_reg(reg_preds)
    reg_std_per_sample = std_reg(reg_preds, mean_reg_pred)

    # 汇总
    out = {
        'model_paths': paths,
        'classification': {
            'vote_mode': args.vote_mode,
            'threshold': float(args.cls_threshold),
            'true': true_cls.tolist(),
            'pred': vote_cls.tolist(),
            'mean_prob': mean_prob_per_sample.tolist()  # 每个样本的平均正类概率
            # 'prob_mean_all': float(np.nanmean(mean_prob_per_sample))  # 全局平均概率
        },
        'regression': {
            'true': true_reg.tolist(),
            'pred': mean_reg_pred.tolist(),
            'std_all_models': reg_std_per_sample.tolist()  # 每个样本的预测标准差
        }
    }

    if not output_path:
        raise ValueError("请提供 --output 或 --main_dir 以确定输出路径")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=4)
    # 控制台简要汇总（仿 predict_elementary.py）
    print('\n' + '=' * 50)
    print('[投票预测完成！关键指标汇总]')
    print('=' * 50)
    # 分类投票（hard/soft）
    acc = float(np.mean(true_cls == vote_cls))
    print('[分类任务]')
    print(f'  投票方式: {args.vote_mode} (threshold={args.cls_threshold:.3f})')
    print(f'  准确率（ACC）: {acc:.4f}')
    print(f'  混淆矩阵:')
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true_cls, vote_cls):
        if t < 2 and p < 2:
            cm[t, p] += 1
    for row in cm:
        print(f'    {row}')

    # 计算召回率（Recall），假设正类别为 1
    # True Positive (TP): cm[1, 1]
    # False Negative (FN): cm[1, 0]
    tp = cm[1, 1]
    fn = cm[1, 0]
    if (tp + fn) > 0:
        recall = float(tp / (tp + fn))
    else:
        # 如果真值为正例的样本数为 0，召回率记为 NaN
        recall = float('nan')
    print(f'  召回率（Recall, 正类 1）: {recall:.4f}')
    # 回归简单平均
    valid_mask = np.isfinite(true_reg)
    if np.any(valid_mask):
        mae = float(np.mean(np.abs(true_reg[valid_mask] - mean_reg_pred[valid_mask])))
        mse = float(np.mean((true_reg[valid_mask] - mean_reg_pred[valid_mask]) ** 2))
        # R2 计算防止除0
        var = np.var(true_reg[valid_mask])
        r2 = float(1 - mse / var) if var > 0 else float('nan')
    else:
        mae = mse = r2 = float('nan')
    print('\n[回归任务]')
    print(f'  MAE: {mae:.4f}')
    print(f'  MSE: {mse:.4f}')
    print(f'  R2: {r2:.4f}')
    print(f'\n结果文件保存路径: {output_path}')
    print('=' * 50)


if __name__ == '__main__':
    main()
