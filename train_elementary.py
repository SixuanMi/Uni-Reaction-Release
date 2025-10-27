import torch
import os
import time
import argparse
import json
import numpy as np

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from utils.data_utils import load_joint_data, fix_seed, count_parameters
from utils.training import train_joint, eval_joint  # 联合训练和评估函数
from utils.Dataset import joint_colfn  # 双标签collate函数

from model import (
    JointModel,  # 联合模型
    RAlignEncoder,
    build_cn_condition_encoder_with_eval  # 保留条件编码器构建函数
)

from rdkit import RDLogger
# 禁用所有 RDKit 日志（包括警告、信息等）
RDLogger.DisableLog('rdApp.*')

def make_dir(args):
    timestamp = time.time()
    detail_dir = os.path.join(args.base_log, f'{timestamp}')
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)
    log_dir = os.path.join(detail_dir, 'log.json')
    best_cls_dir = os.path.join(detail_dir, 'best_cls.pth')  # 最佳分类模型
    best_reg_dir = os.path.join(detail_dir, 'best_reg.pth')  # 最佳回归模型
    return log_dir, best_cls_dir, best_reg_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('联合训练分类和回归任务')
    # 核心参数
    parser.add_argument('--data_path', required=True, type=str, help='数据路径（包含train.csv/val.csv/test.csv）')
    parser.add_argument('--dim', type=int, default=64, help='模型维度')
    parser.add_argument('--heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layer', type=int, default=3, help='编码器层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout概率')
    parser.add_argument('--warmup', type=int, default=20, help='热身轮数')
    parser.add_argument('--lrgamma', type=float, default=1.0, help='学习率衰减系数')
    parser.add_argument('--lr', type=float, default=5e-4, help='初始学习率')
    parser.add_argument('--epoch', type=int, default=200, help='训练总轮数')
    parser.add_argument('--base_log', type=str, default='log_joint', help='日志保存根目录')
    parser.add_argument('--num_worker', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--bs', type=int, default=128, help='批次大小')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='LeakyReLU斜率')
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID（-1为CPU）')
    parser.add_argument('--step_start', type=int, default=0, help='学习率衰减起始轮数')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子（保证可复现）')
    parser.add_argument('--local_heads', type=int, default=4, help='本地注意力头数')
    # 联合训练特有参数
    parser.add_argument('--lambda_reg', type=float, default=0.5, help='回归损失权重λ')
    parser.add_argument('--cls_out_dim', type=int, default=2, help='分类任务输出维度（如2分类）')
    # 条件编码器相关参数（保留，默认不启用）
    parser.add_argument('--use_condition', action='store_true', help='是否启用条件编码器（默认不启用）')
    parser.add_argument('--condition_config', type=str, default='', help='条件编码器配置文件路径（use_condition=True时需提供）')
    parser.add_argument('--condition_both', action='store_true', help='条件是否同时作用于反应物和产物（use_condition=True时生效）')

    args = parser.parse_args()
    print(args)

    # 校验参数：启用条件编码器时必须提供配置文件
    if args.use_condition and not args.condition_config:
        raise ValueError("启用条件编码器时，必须通过--condition_config指定配置文件路径")

    # 固定随机种子
    fix_seed(args.seed)

    # 设备配置
    device = torch.device(f'cuda:{args.device}') if (torch.cuda.is_available() and args.device >= 0) else torch.device('cpu')

    # 加载双任务数据
    train_set, val_set, test_set = load_joint_data(args.data_path)

    # 创建日志目录
    log_dir, best_cls_dir, best_reg_dir = make_dir(args)

    # 数据加载器（使用双标签collate函数）
    train_loader = DataLoader(
        train_set, batch_size=args.bs, shuffle=True,
        collate_fn=joint_colfn, num_workers=args.num_worker,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=args.num_worker,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=joint_colfn, num_workers=args.num_worker,
        pin_memory=True
    )

    # 构建条件信息（根据是否启用条件编码器动态处理）
    if args.use_condition:
        # 启用条件编码器：加载配置并构建条件信息
        with open(args.condition_config) as Fin:
            condition_config = json.load(Fin)
        # 构建条件信息（与原逻辑一致）
        if condition_config['mode'] == 'mix-all':
            condition_infos = {'mixed': {'dim': condition_config['dim'], 'heads': args.heads}}
        elif condition_config['mode'] == 'mix-catalyst-ligand':
            condition_infos = {k: {'dim': condition_config['dim'], 'heads': args.heads}
                              for k in ['additive', 'base', 'catalyst and ligand']}
        else:
            condition_infos = {k: {'dim': condition_config['dim'], 'heads': args.heads}
                              for k in ['ligand', 'base', 'additive', 'catalyst']}
        # 构建条件编码器
        condition_encoder, eval_layers = build_cn_condition_encoder_with_eval(
            config=condition_config, dropout=args.dropout
        )
    else:
        # 不启用条件编码器：条件信息为空，编码器设为None
        condition_infos = {}
        condition_encoder = None

    # 构建基础编码器（根据是否启用条件编码器动态传入参数）
    encoder = RAlignEncoder(
        n_layer=args.n_layer,
        emb_dim=args.dim,
        edge_dim=args.dim,
        heads=args.heads,
        # 条件相关参数：启用时传入实际信息，否则为空
        reac_batch_infos=condition_infos if args.use_condition else {},
        prod_batch_infos=condition_infos if (args.use_condition and args.condition_both) else {},
        prod_num_keys={},
        reac_num_keys={},
        dropout=args.dropout,
        negative_slope=args.negative_slope,
        update_last_edge=False
    )

    # 初始化联合模型（条件编码器可选传入）
    model = JointModel(
        encoder=encoder,
        condition_encoder=condition_encoder,  # 不启用时为None
        dim=args.dim,
        dropout=args.dropout,
        heads=args.heads,
        cls_out_dim=args.cls_out_dim
    ).to(device)

    # 统计模型参数
    total_params, trainable_params = count_parameters(model)
    print(f'总参数: {total_params}, 可训练参数: {trainable_params}')

    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sher = ExponentialLR(optimizer, gamma=args.lrgamma)

    # 日志初始化
    log_info = {
        'args': args.__dict__,
        'train_loss': [],
        'valid_metric': [],
        'test_metric': [],
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout)

    # 跟踪最佳模型
    best_cls_acc, best_cls_ep = -1.0, 0
    best_reg_r2, best_reg_ep = -float('inf'), 0

    # 训练循环
    for ep in range(args.epoch):
        print(f'\n[INFO] 训练轮次 {ep+1}/{args.epoch}')
        # 联合训练（根据是否启用条件编码器决定是否传入has_reag）
        train_loss = train_joint(
            train_loader, model, optimizer, device,
            lambda_reg=args.lambda_reg,
            warmup=(ep < args.warmup),
            total_heads=args.heads,
            local_heads=args.local_heads,
            has_reag=args.use_condition  # 启用条件编码器时为True，否则为False
        )
        # 联合评估（同理）
        val_metric = eval_joint(
            val_loader, model, device,
            total_heads=args.heads,
            local_heads=args.local_heads,
            has_reag=args.use_condition
        )
        test_metric = eval_joint(
            test_loader, model, device,
            total_heads=args.heads,
            local_heads=args.local_heads,
            has_reag=args.use_condition
        )

        # 打印指标
        print(f'[训练损失] {train_loss:.4f}')
        print(f'[验证集] 分类准确率: {val_metric["classification"]["ACC"]:.4f}, '
            f'回归MAE: {val_metric["regression"]["MAE"]:.4f}, '
            f'回归MSE: {val_metric["regression"]["MSE"]:.4f}, '
            f'回归R2: {val_metric["regression"]["R2"]:.4f}')
        print(f'[测试集] 分类准确率: {test_metric["classification"]["ACC"]:.4f}, '
            f'回归MAE: {test_metric["regression"]["MAE"]:.4f}, '
            f'回归MSE: {test_metric["regression"]["MSE"]:.4f}, '
            f'回归R2: {test_metric["regression"]["R2"]:.4f}')

        # 更新日志
        log_info['train_loss'].append(train_loss)
        log_info['valid_metric'].append(val_metric)
        log_info['test_metric'].append(test_metric)
        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        # 学习率衰减
        if ep >= args.warmup and ep >= args.step_start:
            lr_sher.step()
            print(f'[学习率更新] {lr_sher.get_last_lr()[0]:.6f}')

        # 保存最佳模型
        if val_metric["classification"]["ACC"] > best_cls_acc:
            best_cls_acc = val_metric["classification"]["ACC"]
            best_cls_ep = ep + 1
            torch.save(model.state_dict(), best_cls_dir)
            print(f'[最佳分类模型更新] 轮次: {best_cls_ep}, 准确率: {best_cls_acc:.4f}')
        
        if val_metric["regression"]["R2"] > best_reg_r2:
            best_reg_r2 = val_metric["regression"]["R2"]
            best_reg_ep = ep + 1
            torch.save(model.state_dict(), best_reg_dir)
            print(f'[最佳回归模型更新] 轮次: {best_reg_ep}, '
                f'R2: {best_reg_r2:.4f}, '
                f'MAE: {val_metric["regression"]["MAE"]:.4f}, '
                f'MSE: {val_metric["regression"]["MSE"]:.4f}')

    # 输出最终结果
    print('\n[训练完成]')
    print(f'最佳分类模型：轮次 {best_cls_ep}, 验证准确率 {best_cls_acc:.4f}')
    print(f'最佳回归模型：轮次 {best_reg_ep}, 验证R2 {best_reg_r2:.4f}, '
        f'验证MAE {val_metric["regression"]["MAE"]:.4f}, '
        f'验证MSE {val_metric["regression"]["MSE"]:.4f}')
    
    # # H200 训练结束后不会正常退出，尝试强制终止
    # import threading
    
    # main_thread = threading.current_thread()
    # for thread in threading.enumerate():
    #     if thread is not main_thread and not thread.daemon:
    #         print(f"强制终止非守护线程: {thread.name}")
    #         # 若线程有stop方法，优先调用
    #         if hasattr(thread, 'stop'):
    #             thread.stop()
    #         # 等待线程终止（最多5秒）
    #         thread.join(timeout=5.0)

    # if 'train_loader' in locals():
    #     del train_loader  # 删除DataLoader对象
    # if 'val_loader' in locals():
    #     del val_loader
    # if 'test_loader' in locals():
    #     del test_loader

    # # 强制清理CUDA相关资源
    # torch.cuda.empty_cache()  # 清空CUDA缓存
    # torch.cuda.reset_max_memory_allocated()  # 重置内存统计