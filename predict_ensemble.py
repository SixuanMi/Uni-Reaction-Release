import argparse
import glob
import json
import os
from typing import List

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


def majority_vote_cls(cls_preds: np.ndarray) -> np.ndarray:
    # cls_preds: [n_models, n_samples]
    out = []
    for i in range(cls_preds.shape[1]):
        votes = np.bincount(cls_preds[:, i].astype(int))
        out.append(np.argmax(votes))
    return np.array(out)


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
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--local_heads', type=int, default=4)
    parser.add_argument(
        '--fusion_mode', type=str, default='legacy', choices=['legacy', 'film'],
        help='R/P融合方式（需与训练一致）'
    )
    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)
    device = resolve_device(args.device)

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
    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=args.num_worker, pin_memory=True
    )

    # 逐模型预测
    cls_preds = []
    cls_scores = []
    reg_preds = []
    true_cls = None
    true_reg = None

    for p in paths:
        m = build_joint_model(args, dropout=0.0).to(device)
        state = torch.load(p, map_location=device)
        m.load_state_dict(state)
        m.eval()
        print(f"[INFO] 模型加载完成: {p}")

        res = eval_joint(
            loader=test_loader,
            model=m,
            device=device,
            total_heads=args.heads,
            local_heads=args.local_heads,
            return_raw=True,
            pos_label=1,
            lambda_reg=0.005
        )
        cls_preds.append(np.array(res['raw']['cls_pred']))
        cls_scores.append(np.array(res['raw']['cls_scores']))
        reg_preds.append(np.array(res['raw']['reg_pred']))
        if true_cls is None:
            true_cls = np.array(res['raw']['cls_true'])
            true_reg = np.array(res['raw']['reg_true'])

    cls_preds = np.stack(cls_preds, axis=0)
    cls_scores = np.stack(cls_scores, axis=0)
    reg_preds = np.stack(reg_preds, axis=0)

    # 集成
    vote_cls = majority_vote_cls(cls_preds)
    mean_reg_pred = mean_reg(reg_preds)

    mean_prob_per_sample = np.nanmean(cls_scores, axis=0)  # [n_samples]
    reg_std_per_sample = std_reg(reg_preds, mean_reg_pred)

    # 汇总
    out = {
        'model_paths': paths,
        'classification': {
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
    # 分类多数投票
    acc = float(np.mean(true_cls == vote_cls))
    print('[分类任务]')
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
