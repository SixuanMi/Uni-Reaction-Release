import torch
import os
import argparse
import json
import numpy as np

from torch.utils.data import DataLoader

from utils.data_utils import load_joint_data, fix_seed
from utils.model_factory import build_joint_model, resolve_device
from utils.training import eval_joint  # 复用联合评估函数（已支持新增分类指标）
from utils.Dataset import joint_colfn

from rdkit import RDLogger
# 禁用RDKit日志
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('联合模型预测（适配FocalLoss+完整分类指标）')
    # 核心参数（与训练脚本完全保持一致）
    parser.add_argument('--data_path', required=True, type=str, help='数据路径（包含test.csv）')
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
        '--fusion_mode', type=str, default='legacy', choices=['legacy', 'film'],
        help='R/P融合方式：legacy为原始对齐融合，film为对称共享FiLM（需与训练一致）'
    )
    parser.add_argument('--output_path', required=True, type=str, help='输出结果保存路径（.json）')
    parser.add_argument('--checkpoint', required=True, type=str, help='模型权重文件路径（.pth）')
    # 任务相关参数
    parser.add_argument('--pos_label', type=int, default=1, help='正类标签（需与训练一致，默认1，即"实际为真"的标签）')

    args = parser.parse_args()
    print(args)

    # 参数校验
    if args.pos_label != 1:
        raise ValueError('当前固定为二分类，正类标签必须为 1')

    # 固定随机种子
    fix_seed(args.seed)

    # 设备配置
    device = resolve_device(args.device)

    # 加载数据（仅测试集，复用训练时的数据加载逻辑）
    _, _, test_set = load_joint_data(args.data_path)

    # 数据加载器（保持与训练一致的collate_fn）
    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=args.num_worker,
        pin_memory=True
    )

    model = build_joint_model(args, dropout=0.0).to(device)

    # 加载模型权重（支持CPU/GPU自动适配）
    print(f'[INFO] 加载模型权重: {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式（关键：禁用BatchNorm/ dropout）
    print(f'[INFO] 模型加载完成，设备：{device}')

    # 执行预测
    results = eval_joint(
        test_loader, model, device,
        lambda_reg=0.005,
        total_heads=args.heads,
        local_heads=args.local_heads,
        pos_label=args.pos_label,
        return_raw=True
    )

    # 整理输出结果（包含所有新增分类指标）
    output = {
        'metadata': {
            'checkpoint': args.checkpoint,
            'dim': args.dim,
            'n_layer': args.n_layer,
            'pos_label': args.pos_label,
            'batch_size': args.bs,
            'fusion_mode': args.fusion_mode
        },
        'classification': {
            'true_labels': results['raw']['cls_true'],  # 分类真实标签（列表）
            'pred_labels': results['raw']['cls_pred'],  # 分类预测标签（列表）
            'cls_scores': results['raw']['cls_scores'],  # 新增：正类置信度得分（AUC计算必需）
            'metrics': {
                'ACC': results['classification']['ACC'],
                'Precision': results['classification']['Precision'],  # 精确率
                'Recall': results['classification']['Recall'],        # 召回率
                'F1': results['classification']['F1'],                # F1分数
                'Confusion_Matrix': results['classification']['Confusion_Matrix']  # 混淆矩阵
            }
        },
        'regression': {
            'true_values': results['raw']['reg_true'],  # 回归真实值（含NaN，与分类标签长度一致）
            'pred_values': results['raw']['reg_pred'],  # 回归预测值（含NaN）
            'metrics': {
                'MAE': results['regression']['MAE'],
                'MSE': results['regression']['MSE'],
                'R2': results['regression']['R2']
            }
        }
    }

    # 确保输出目录存在（避免路径不存在报错）
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存完整结果（缩进4格，便于阅读）
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=4)

    # 打印关键结果）
    print('\n' + '='*50)
    print('[预测完成！关键指标汇总]')
    print('='*50)
    # 分类指标（重点展示漏检率）
    print(f'[分类任务]')
    print(f'  准确率（ACC）: {output["classification"]["metrics"]["ACC"]:.4f}')
    print(f'  精确率（Precision）: {output["classification"]["metrics"]["Precision"]:.4f}')
    print(f'  召回率（Recall）: {output["classification"]["metrics"]["Recall"]:.4f}')
    print(f'  F1分数: {output["classification"]["metrics"]["F1"]:.4f}')
    print(f'  混淆矩阵:')
    cm = np.array(output["classification"]["metrics"]["Confusion_Matrix"])
    for row in cm:
        print(f'    {row}')
    # 回归指标
    print(f'\n[回归任务]')
    print(f'  MAE: {output["regression"]["metrics"]["MAE"]:.4f}')
    print(f'  MSE: {output["regression"]["metrics"]["MSE"]:.4f}')
    print(f'  R2: {output["regression"]["metrics"]["R2"]:.4f}')
    print(f'\n结果文件保存路径: {args.output_path}')
    print('='*50)
