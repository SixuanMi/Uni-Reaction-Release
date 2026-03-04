import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.Dataset import JointDataset, joint_colfn
from utils.data_utils import fix_seed
from utils.model_factory import build_joint_model, resolve_device
from utils.tensor_utils import generate_local_global_mask

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def swap_reaction(reaction: str) -> str:
    reactants, products = reaction.strip().split('>>')
    return f'{products}>>{reactants}'


def load_joint_split(data_path, part):
    csv_path = os.path.join(data_path, f'{part}.csv')
    data = pd.read_csv(csv_path)

    reactions = data['Reaction'].tolist()
    cls_labels = data['Is_elementary'].astype(int).tolist()
    reg_labels = [
        float(x) if pd.notna(x) else float('nan')
        for x in data['Barrier'].tolist()
    ]

    orig_set = JointDataset(
        reactions=reactions,
        is_elementary=cls_labels,
        barrier=reg_labels
    )
    swapped_set = JointDataset(
        reactions=[swap_reaction(x) for x in reactions],
        is_elementary=cls_labels,
        barrier=reg_labels
    )
    return reactions, cls_labels, reg_labels, orig_set, swapped_set


def build_model(args, device):
    model = build_joint_model(args, dropout=0.0).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_predictions(loader, model, device, total_heads, local_heads, pos_label):
    cls_true, reg_true = [], []
    cls_pred, cls_scores, reg_pred = [], [], []

    for batch_data in loader:
        reac, prod, cls_label, reg_label = batch_data
        reac = reac.to(device)
        prod = prod.to(device)
        cls_label = cls_label.to(device)
        reg_label = reg_label.to(device)

        if local_heads > 0:
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None

        with torch.no_grad():
            cls_out, reg_out = model(reac, prod, None, cross_mask=cross_mask)

        cls_true.append(cls_label.cpu().numpy())
        reg_true.append(reg_label.cpu().numpy())
        cls_pred.append(cls_out.argmax(dim=1).cpu().numpy())
        cls_scores.append(torch.softmax(cls_out, dim=1)[:, pos_label].cpu().numpy())
        reg_pred.append(reg_out.view(-1).cpu().numpy())

    return {
        'cls_true': np.concatenate(cls_true, axis=0),
        'reg_true': np.concatenate(reg_true, axis=0),
        'cls_pred': np.concatenate(cls_pred, axis=0),
        'cls_scores': np.concatenate(cls_scores, axis=0),
        'reg_pred': np.concatenate(reg_pred, axis=0),
    }


def summarize_abs(x):
    abs_x = np.abs(x)
    return {
        'mean_abs': float(np.mean(abs_x)),
        'median_abs': float(np.median(abs_x)),
        'p90_abs': float(np.percentile(abs_x, 90)),
        'p95_abs': float(np.percentile(abs_x, 95)),
        'max_abs': float(np.max(abs_x)),
    }


def build_top_cases(reactions, delta, orig_vals, swap_vals, top_k):
    abs_delta = np.abs(delta)
    top_idx = np.argsort(abs_delta)[::-1][:top_k]
    cases = []
    for idx in top_idx.tolist():
        cases.append({
            'index': int(idx),
            'csv_row': int(idx + 2),
            'reaction': reactions[idx],
            'orig': float(orig_vals[idx]),
            'swapped': float(swap_vals[idx]),
            'delta': float(delta[idx]),
            'abs_delta': float(abs_delta[idx]),
        })
    return cases


def nan_summary():
    return {
        'mean_abs': float('nan'),
        'median_abs': float('nan'),
        'p90_abs': float('nan'),
        'p95_abs': float('nan'),
        'max_abs': float('nan'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('交换R/P敏感性诊断（JointModel）')
    parser.add_argument('--data_path', required=True, type=str, help='数据路径（包含train/val/test.csv）')
    parser.add_argument('--part', type=str, default='test', choices=['train', 'val', 'test'], help='要评估的数据切分')
    parser.add_argument('--checkpoint', required=True, type=str, help='模型权重文件路径（.pth）')
    parser.add_argument('--output_path', required=True, type=str, help='输出结果保存路径（.json）')
    parser.add_argument('--dim', type=int, default=128, help='模型维度（需与训练一致）')
    parser.add_argument('--heads', type=int, default=8, help='注意力头数（需与训练一致）')
    parser.add_argument('--n_layer', type=int, default=3, help='编码器层数（需与训练一致）')
    parser.add_argument('--num_worker', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--bs', type=int, default=128, help='批次大小')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='LeakyReLU斜率（需与训练一致）')
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID（-1为CPU）')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子')
    parser.add_argument('--local_heads', type=int, default=4, help='本地注意力头数（需与训练一致）')
    parser.add_argument(
        '--share_reac_prod_encoder',
        dest='share_reac_prod_encoder',
        action='store_true',
        default=True,
        help='反应物/产物编码层共享参数（默认开启）'
    )
    parser.add_argument(
        '--no_share_reac_prod_encoder',
        dest='share_reac_prod_encoder',
        action='store_false',
        help='关闭反应物/产物编码层共享参数'
    )
    parser.add_argument(
        '--fusion_mode', type=str, default='legacy', choices=['legacy', 'film'],
        help='R/P融合方式（需与训练一致）'
    )
    parser.add_argument('--pos_label', type=int, default=1, help='正类标签（默认1）')
    parser.add_argument('--top_k', type=int, default=20, help='输出变化最大的前K条样本')

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)
    device = resolve_device(args.device)

    reactions, cls_labels, reg_labels, orig_set, swapped_set = load_joint_split(
        args.data_path, args.part
    )
    orig_loader = DataLoader(
        orig_set, batch_size=args.bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=args.num_worker, pin_memory=True
    )
    swapped_loader = DataLoader(
        swapped_set, batch_size=args.bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=args.num_worker, pin_memory=True
    )

    model = build_model(args, device)

    print('[INFO] 开始原始方向预测')
    orig_pred = collect_predictions(
        loader=orig_loader,
        model=model,
        device=device,
        total_heads=args.heads,
        local_heads=args.local_heads,
        pos_label=args.pos_label
    )
    print('[INFO] 开始交换方向预测')
    swapped_pred = collect_predictions(
        loader=swapped_loader,
        model=model,
        device=device,
        total_heads=args.heads,
        local_heads=args.local_heads,
        pos_label=args.pos_label
    )

    cls_prob_delta = swapped_pred['cls_scores'] - orig_pred['cls_scores']
    cls_flip = (orig_pred['cls_pred'] != swapped_pred['cls_pred']).astype(np.float32)
    reg_delta = swapped_pred['reg_pred'] - orig_pred['reg_pred']

    reg_true = np.array(reg_labels, dtype=np.float64)
    reg_labeled_mask = np.isfinite(reg_true)
    reg_labeled_delta = reg_delta[reg_labeled_mask]
    reg_labeled_summary = summarize_abs(reg_labeled_delta) \
        if reg_labeled_delta.size > 0 else nan_summary()

    result = {
        'metadata': {
            'checkpoint': args.checkpoint,
            'data_path': args.data_path,
            'part': args.part,
            'batch_size': args.bs,
            'dim': args.dim,
            'heads': args.heads,
            'n_layer': args.n_layer,
            'fusion_mode': args.fusion_mode,
            'num_samples': len(reactions),
            'num_reg_labeled': int(reg_labeled_mask.sum()),
        },
        'classification': {
            'orig_acc': float(np.mean(orig_pred['cls_true'] == orig_pred['cls_pred'])),
            'swapped_acc': float(np.mean(orig_pred['cls_true'] == swapped_pred['cls_pred'])),
            'pred_flip_rate': float(np.mean(cls_flip)),
            'score_delta': {
                'mean_signed': float(np.mean(cls_prob_delta)),
                **summarize_abs(cls_prob_delta)
            },
            'top_score_shift_cases': build_top_cases(
                reactions=reactions,
                delta=cls_prob_delta,
                orig_vals=orig_pred['cls_scores'],
                swap_vals=swapped_pred['cls_scores'],
                top_k=args.top_k
            )
        },
        'regression': {
            'all_samples': {
                'mean_signed': float(np.mean(reg_delta)),
                **summarize_abs(reg_delta)
            },
            'labeled_only': {
                'mean_signed': float(np.mean(reg_labeled_delta)) if reg_labeled_delta.size > 0 else float('nan'),
                **reg_labeled_summary
            },
            'top_reg_shift_cases': build_top_cases(
                reactions=reactions,
                delta=reg_delta,
                orig_vals=orig_pred['reg_pred'],
                swap_vals=swapped_pred['reg_pred'],
                top_k=args.top_k
            )
        }
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, 'w') as fout:
        json.dump(result, fout, indent=4)

    print('\n' + '=' * 50)
    print('[交换敏感性诊断完成]')
    print('=' * 50)
    print('[分类]')
    print(f'  原始ACC: {result["classification"]["orig_acc"]:.4f}')
    print(f'  交换后ACC: {result["classification"]["swapped_acc"]:.4f}')
    print(f'  预测翻转率: {result["classification"]["pred_flip_rate"]:.4f}')
    print(f'  正类概率平均绝对变化: {result["classification"]["score_delta"]["mean_abs"]:.6f}')
    print(f'  正类概率95分位绝对变化: {result["classification"]["score_delta"]["p95_abs"]:.6f}')
    print('[回归]')
    print(f'  全样本预测平均绝对变化: {result["regression"]["all_samples"]["mean_abs"]:.6f}')
    print(f'  有标签样本预测平均绝对变化: {result["regression"]["labeled_only"]["mean_abs"]:.6f}')
    print(f'  有标签样本预测95分位绝对变化: {result["regression"]["labeled_only"]["p95_abs"]:.6f}')
    print(f'\n结果文件保存路径: {args.output_path}')
    print('=' * 50)
