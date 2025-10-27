import torch
import os
import argparse
import json
import numpy as np

from torch.utils.data import DataLoader

from utils.data_utils import load_joint_data, fix_seed
from utils.training import eval_joint  # 复用联合评估函数
from utils.Dataset import joint_colfn

from model import (
    JointModel, RAlignEncoder, build_cn_condition_encoder_with_eval
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('联合模型预测')
    # 原有参数
    parser.add_argument('--data_path', required=True, type=str, help='数据路径')
    parser.add_argument('--dim', type=int, default=128, help='模型维度')
    parser.add_argument('--heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layer', type=int, default=3, help='层数')
    parser.add_argument('--num_worker', type=int, default=8, help='数据加载线程')
    parser.add_argument('--bs', type=int, default=128, help='批次大小')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='LeakyReLU斜率')
    parser.add_argument('--device', type=int, default=0, help='设备ID')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子')
    parser.add_argument('--condition_config', required=True, type=str, help='条件编码器配置')
    parser.add_argument('--condition_both', action='store_true', help='条件同时作用于反应物和产物')
    parser.add_argument('--local_heads', type=int, default=4, help='本地注意力头数')
    parser.add_argument('--output_path', required=True, type=str, help='输出结果路径')
    parser.add_argument('--checkpoint', required=True, type=str, help='模型 checkpoint')
    # 新增参数：预测任务类型
    parser.add_argument('--task', required=True, type=str, choices=['classification', 'regression'], help='预测任务类型')
    # 分类任务输出维度（需与训练时一致）
    parser.add_argument('--cls_out_dim', type=int, default=2, help='分类任务输出维度')

    args = parser.parse_args()

    # 设备设置
    device = torch.device(f'cuda:{args.device}') if (torch.cuda.is_available() and args.device >= 0) else torch.device('cpu')

    # 加载条件配置
    with open(args.condition_config) as Fin:
        condition_config = json.load(Fin)

    # 加载数据（仅测试集）
    _, _, test_set = load_cn_joint_data(
        args.data_path, condition_config['data_type']
    )

    # 数据加载器
    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=args.num_worker
    )

    # 构建条件信息（与训练一致）
    if condition_config['mode'] == 'mix-all':
        condition_infos = {'mixed': {'dim': condition_config['dim'], 'heads': args.heads}}
    elif condition_config['mode'] == 'mix-catalyst-ligand':
        condition_infos = {k: {'dim': condition_config['dim'], 'heads': args.heads}
                          for k in ['additive', 'base', 'catalyst and ligand']}
    else:
        condition_infos = {k: {'dim': condition_config['dim'], 'heads': args.heads}
                          for k in ['ligand', 'base', 'additive', 'catalyst']}

    # 构建编码器
    encoder = RAlignEncoder(
        n_layer=args.n_layer, emb_dim=args.dim, edge_dim=args.dim,
        heads=args.heads, reac_batch_infos=condition_infos,
        prod_batch_infos=condition_infos if args.condition_both else {},
        prod_num_keys={}, reac_num_keys={}, dropout=0,  # 预测时不使用dropout
        negative_slope=args.negative_slope, update_last_edge=False
    )

    # 构建条件编码器
    condition_encoder, _ = build_cn_condition_encoder_with_eval(
        config=condition_config, dropout=0  # 预测时不使用dropout
    )

    # 初始化联合模型
    model = JointModel(
        encoder=encoder,
        condition_encoder=condition_encoder,
        dim=args.dim,
        dropout=0,
        heads=args.heads,
        cls_out_dim=args.cls_out_dim
    ).to(device)

    # 加载模型权重
    print(f'[INFO] 加载模型: {args.checkpoint}')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 执行预测（复用eval_joint获取原始结果）
    results = eval_joint(
        test_loader, model, device,
        total_heads=args.heads,
        local_heads=args.local_heads,
        return_raw=True
    )

    # 根据任务类型提取结果
    output = {
        'task': args.task,
        'true': results['raw'][f'{args.task}_true'],
        'pred': results['raw'][f'{args.task}_pred'],
        'metrics': results[args.task]
    }

    # 保存结果
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=4)

    # 打印指标
    if args.task == 'classification':
        print(f'分类准确率: {output["metrics"]["ACC"]:.4f}')
    else:
        print(f'回归MAE: {output["metrics"]["MAE"]:.4f}')
        print(f'回归MSE: {output["metrics"]["MSE"]:.4f}')
        print(f'回归R2: {output["metrics"]["R2"]:.4f}')