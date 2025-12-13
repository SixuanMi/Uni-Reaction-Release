import argparse
import glob
import json
import os
from typing import List

import numpy as np
import torch

from utils.data_utils import load_joint_data, fix_seed
from utils.training.training import eval_joint
from utils.Dataset import joint_colfn
from model import JointModel, RAlignEncoder, build_cn_condition_encoder_with_eval
from torch.utils.data import DataLoader

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def collect_model_paths(model_dir: str, model_list: str = None, model_paths: List[str] = None) -> List[str]:
    if model_paths:
        return model_paths
    if model_list:
        with open(model_list) as f:
            return [ln.strip() for ln in f if ln.strip()]
    if model_dir:
        paths = glob.glob(os.path.join(model_dir, "**", "best_loss.pth"), recursive=True)
        if not paths:
            paths = glob.glob(os.path.join(model_dir, "**", "best_model.pt"), recursive=True)
        return sorted(paths)
    raise ValueError("需要提供 model_dir / model_list / model_paths 之一")


def build_model(args, dropout: float):
    if args.use_condition:
        with open(args.condition_config) as fin:
            condition_config = json.load(fin)
        if condition_config['mode'] == 'mix-all':
            condition_infos = {'mixed': {'dim': condition_config['dim'], 'heads': args.heads}}
        elif condition_config['mode'] == 'mix-catalyst-ligand':
            condition_infos = {
                k: {'dim': condition_config['dim'], 'heads': args.heads}
                for k in ['additive', 'base', 'catalyst and ligand']
            }
        else:
            condition_infos = {
                k: {'dim': condition_config['dim'], 'heads': args.heads}
                for k in ['ligand', 'base', 'additive', 'catalyst']
            }
        condition_encoder, _ = build_cn_condition_encoder_with_eval(
            config=condition_config, dropout=dropout
        )
    else:
        condition_infos = {}
        condition_encoder = None

    encoder = RAlignEncoder(
        n_layer=args.n_layer,
        emb_dim=args.dim,
        edge_dim=args.dim,
        heads=args.heads,
        reac_batch_infos=condition_infos if args.use_condition else {},
        prod_batch_infos=condition_infos if (args.use_condition and args.condition_both) else {},
        prod_num_keys={},
        reac_num_keys={},
        dropout=dropout,
        negative_slope=args.negative_slope,
        update_last_edge=False
    )

    return JointModel(
        encoder=encoder,
        condition_encoder=condition_encoder,
        dim=args.dim,
        dropout=dropout,
        heads=args.heads,
        cls_out_dim=args.cls_out_dim
    )


def majority_vote_cls(cls_preds: np.ndarray) -> np.ndarray:
    # cls_preds: [n_models, n_samples]
    out = []
    for i in range(cls_preds.shape[1]):
        votes = np.bincount(cls_preds[:, i].astype(int))
        out.append(np.argmax(votes))
    return np.array(out)


def mean_reg(all_reg: np.ndarray) -> np.ndarray:
    # all_reg: [n_models, n_samples]
    with np.errstate(all='ignore'):
        out = np.nanmean(all_reg, axis=0)
    # 如果某个样本所有模型均为 NaN，显式设为 NaN，避免警告/inf
    all_nan = np.all(~np.isfinite(all_reg), axis=0)
    out[all_nan] = np.nan
    return out


def main():
    parser = argparse.ArgumentParser("多模型投票/平均评估")
    parser.add_argument('--data_path', required=True, help='包含 test.csv 的目录')
    parser.add_argument('--model_dir', type=str, default=None, help='递归查找 best_loss.pth/best_model.pt 的目录')
    parser.add_argument('--model_list', type=str, default=None, help='txt，一行一个模型路径')
    parser.add_argument('--model_paths', nargs='+', default=None, help='直接提供模型路径列表')
    parser.add_argument('--output', required=True, help='输出结果 json')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--cls_out_dim', type=int, default=2)
    parser.add_argument('--local_heads', type=int, default=4)
    parser.add_argument('--use_condition', action='store_true')
    parser.add_argument('--condition_config', type=str, default='')
    parser.add_argument('--condition_both', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.use_condition and not args.condition_config:
        raise ValueError("启用条件编码器需提供 --condition_config")

    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}') if (torch.cuda.is_available() and args.device >= 0) else torch.device('cpu')

    paths = collect_model_paths(args.model_dir, args.model_list, args.model_paths)
    if len(paths) == 0:
        raise ValueError("未找到任何模型")
    print(f"[INFO] 模型数: {len(paths)}")

    # 加载测试集
    _, _, test_set = load_joint_data(args.data_path)
    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=args.num_worker, pin_memory=True
    )

    # 逐模型预测
    cls_preds = []
    reg_preds = []
    true_cls = None
    true_reg = None

    for p in paths:
        m = build_model(args, dropout=0.0).to(device)
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
            has_reag=False,
            num_classes=args.cls_out_dim,
            pos_label=1,
            lambda_reg=0.005
        )
        cls_preds.append(np.array(res['raw']['cls_pred']))
        reg_preds.append(np.array(res['raw']['reg_pred']))
        if true_cls is None:
            true_cls = np.array(res['raw']['cls_true'])
            true_reg = np.array(res['raw']['reg_true'])

    cls_preds = np.stack(cls_preds, axis=0)
    reg_preds = np.stack(reg_preds, axis=0)

    # 集成
    vote_cls = majority_vote_cls(cls_preds)
    mean_reg_pred = mean_reg(reg_preds)

    # 汇总
    out = {
        'model_paths': paths,
        'classification': {
            'true': true_cls.tolist(),
            'pred': vote_cls.tolist()
        },
        'regression': {
            'true': true_reg.tolist(),
            'pred': mean_reg_pred.tolist()
        }
    }

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
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
    cm = np.zeros((args.cls_out_dim, args.cls_out_dim), dtype=int)
    for t, p in zip(true_cls, vote_cls):
        if t < args.cls_out_dim and p < args.cls_out_dim:
            cm[t, p] += 1
    for row in cm:
        print(f'    {row}')
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
    print(f'\n结果文件保存路径: {args.output}')
    print('=' * 50)


if __name__ == '__main__':
    main()
