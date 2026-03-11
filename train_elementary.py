import torch
import os
import time
import argparse
import json
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.data_utils import load_joint_data, fix_seed, count_parameters
from utils.model_factory import build_joint_model, resolve_device
from utils.training import train_joint, eval_joint  # 联合训练和评估函数
from utils.training.training import warmup_lr_scheduler
from utils.Dataset import joint_colfn  # 双标签collate函数

from rdkit import RDLogger
# 禁用所有 RDKit 日志（包括警告、信息等）
RDLogger.DisableLog('rdApp.*')

def make_dir(args):
    timestamp = time.time()
    detail_dir = os.path.join(args.base_log, f'{timestamp}')
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)
    log_dir = os.path.join(detail_dir, 'log.json')
    best_model_dir = os.path.join(detail_dir, 'best_model.pth')
    return log_dir, best_model_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('联合训练分类和回归任务')
    # 核心参数
    parser.add_argument('--data_path', required=True, type=str, help='数据路径（包含train.csv/val.csv/test.csv）')
    parser.add_argument('--dim', type=int, default=192, help='模型维度') # 192
    parser.add_argument('--heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layer', type=int, default=5, help='编码器层数') # 5
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout概率')
    parser.add_argument('--warmup', type=int, default=5, help='热身轮数')
    parser.add_argument('--lrfactor', type=float, default=0.5, help='学习率衰减系数') # 0.7
    parser.add_argument('--lrpatience', type=int, default=5, help='验证集指标连续未衰减轮数')
    parser.add_argument('--auc_delta', type=float, default=2e-5, help='PR-AUC最小提升阈值（用于LR调度/最佳模型/早停）')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='学习率最小值（用于LR调度与早停触发）')
    parser.add_argument('--lr', type=float, default=2e-4, help='初始学习率') # 2e-4
    parser.add_argument('--epoch', type=int, default=200, help='训练总轮数') # 100-200
    parser.add_argument('--base_log', type=str, default='log_joint', help='日志保存根目录')
    parser.add_argument('--num_worker', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--bs', type=int, default=128, help='批次大小')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='LeakyReLU斜率')
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID（-1为CPU）')
    parser.add_argument('--step_start', type=int, default=20, help='学习率衰减起始轮数')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子（保证可复现）')
    parser.add_argument('--local_heads', type=int, default=4, help='本地注意力头数')
    parser.add_argument(
        '--fusion_mode', type=str, default='film', choices=['legacy', 'film'],
        help='R/P融合方式：legacy为原始对齐融合，film为对称共享FiLM'
    )
    parser.add_argument('--lambda_reg', type=float, default=5e-3, help='回归损失权重 λ')
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=None,
        help='早停轮数（默认lrpatience*2，仅在学习率到达最小值后开始计数）'
    )

    args = parser.parse_args()
    print(args)

    # 固定随机种子
    fix_seed(args.seed)

    # 设备配置
    device = resolve_device(args.device)

    current_lambda = args.lambda_reg

    # 早停参数：默认等于学习率衰减耐心轮数的2倍
    if args.early_stop_patience is None:
        early_stop_patience = args.lrpatience * 2
    else:
        early_stop_patience = args.early_stop_patience
    if early_stop_patience < 0:
        raise ValueError("--early_stop_patience 不能为负数")
    if args.min_lr <= 0:
        raise ValueError("--min_lr 必须大于0")
    if args.auc_delta < 0:
        raise ValueError("--auc_delta 不能为负数")
        
    # 加载双任务数据
    train_set, val_set, test_set = load_joint_data(args.data_path)

    # 创建日志目录
    log_dir, best_model_dir = make_dir(args)

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

    model = build_joint_model(args, dropout=args.dropout).to(device)

    # 统计模型参数
    total_params, trainable_params = count_parameters(model)
    print(f'总参数: {total_params}, 可训练参数: {trainable_params}')

    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sher = ReduceLROnPlateau(
        optimizer,
        mode='max',             # 匹配PR-AUC：越大越好
        factor=args.lrfactor,   # 衰减系数（每次衰减为原来百分比）
        patience=args.lrpatience,  # 验证集指标连续未提升轮数达到阈值后衰减
        threshold=args.auc_delta,
        threshold_mode='abs',
        min_lr=args.min_lr,
        # min_lr=1e-5,            # 最小学习率（避免衰减到0）
    )
    print(
        f'[早停设置] min_lr={args.min_lr:.6f}, '
        f'early_stop_patience={early_stop_patience}, '
        f'metric=val PR-AUC, auc_delta={args.auc_delta:.1e}'
    )
    warmup_total_steps = max(args.warmup, 0) * len(train_loader)
    warmup_sher = None
    if warmup_total_steps > 0:
        warmup_sher = warmup_lr_scheduler(
            optimizer, warmup_total_steps, 5e-2
        )
        print(f'[Warmup设置] warmup_epochs={args.warmup}, warmup_steps={warmup_total_steps}')
    else:
        print('[Warmup设置] warmup关闭')


    # 日志初始化
    log_info = {
        'args': args.__dict__,
        'early_stop_patience': early_stop_patience,
        'early_stop_min_lr': args.min_lr,
        'auc_delta': args.auc_delta,
        'early_stop_epoch': None,
        'train_total_loss': [],
        'train_cls_loss': [],
        'train_reg_loss': [],
        'current_lambda': [],
        'current_lr': [],
        'valid_metric': [],
        'test_metric': [],
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout)

    # 跟踪最佳模型（按验证集PR-AUC保存）
    best_cls_pr_auc, best_pr_auc_ep = -float('inf'), 0
    min_lr_no_improve_epochs = 0
    min_lr_best_pr_auc = -float('inf')
    early_stop_epoch = None

    # 训练循环
    for ep in range(args.epoch):
        print(f'\n[INFO] 训练轮次 {ep+1}/{args.epoch}')
        current_lr = optimizer.param_groups[0]['lr']  # ReduceLROnPlateau用这个获取LR
        print(f'[学习率状态] 当前学习率：{current_lr:.6f}')
        
        # 联合训练
        # 接收总损失、分类损失、回归损失
        train_total_loss, train_cls_loss, train_reg_loss = train_joint(
            train_loader, model, optimizer, device,
            lambda_reg=current_lambda,
            total_heads=args.heads,
            local_heads=args.local_heads,
            warmup_scheduler=warmup_sher,
            warmup_total_steps=warmup_total_steps
        )
        # 联合评估
        val_metric = eval_joint(
            val_loader, model, device,
            lambda_reg=current_lambda,
            total_heads=args.heads,
            local_heads=args.local_heads,
            pos_label=1
        )
        test_metric = eval_joint(
            test_loader, model, device,
            total_heads=args.heads,
            local_heads=args.local_heads,
            pos_label=1
        )

        # 打印指标
        print(f'[训练总损失]: {train_total_loss:.4f}, [分类损失]: {train_cls_loss:.4f}, [回归损失]: {train_reg_loss:.4f}, [加权后回归损失]: {current_lambda * train_reg_loss:.4f}')
        print('----------')
        # print(f'[验证集] 分类指标：'
        #     f'ACC: {val_metric["classification"]["ACC"]:.4f}, '
        #     f'精确率: {val_metric["classification"]["Precision"]:.4f}, '
        #     f'召回率: {val_metric["classification"]["Recall"]:.4f}, '
        #     f'F1: {val_metric["classification"]["F1"]:.4f}, ')
        print(f'[验证总损失]: {val_metric["validation_loss"]["total_loss"]:.4f}')
        print(
            f'[验证集] ACC: {val_metric["classification"]["ACC"]:.4f}, '
            f'F1: {val_metric["classification"]["F1"]:.4f}, '
            f'PR-AUC: {val_metric["classification"]["PR_AUC"]:.5f}, '
            f'PR阈值(max-F1): {val_metric["classification"]["PR_BEST_F1_THRESHOLD"]:.4f}, '
            f'PR-maxF1: {val_metric["classification"]["PR_BEST_F1"]:.4f}'
        )
        print(f'[验证集] 分类混淆矩阵：\n{np.array(val_metric["classification"]["Confusion_Matrix"])}')
        print(f'[验证集] 回归 MAE: {val_metric["regression"]["MAE"]:.4f}, MSE: {val_metric["regression"]["MSE"]:.4f}, R2: {val_metric["regression"]["R2"]:.4f}')
        print('----------')
        
        # 测试集打印同理
        # print(f'[测试集] 分类指标：'
        #     f'ACC: {test_metric["classification"]["ACC"]:.4f}, '
        #     f'精确率: {test_metric["classification"]["Precision"]:.4f}, '
        #     f'召回率: {test_metric["classification"]["Recall"]:.4f}, '
        #     f'F1: {test_metric["classification"]["F1"]:.4f}, ')
        print(
            f'[测试集] ACC: {test_metric["classification"]["ACC"]:.4f}, '
            f'F1: {test_metric["classification"]["F1"]:.4f}, '
            f'PR-AUC: {test_metric["classification"]["PR_AUC"]:.5f}, '
            f'PR阈值(max-F1): {test_metric["classification"]["PR_BEST_F1_THRESHOLD"]:.4f}, '
            f'PR-maxF1: {test_metric["classification"]["PR_BEST_F1"]:.4f}'
        )
        print(f'[测试集] 分类混淆矩阵：\n{np.array(test_metric["classification"]["Confusion_Matrix"])}')
        print(f'[测试集] 回归 MAE: {test_metric["regression"]["MAE"]:.4f}, MSE: {test_metric["regression"]["MSE"]:.4f}, R2: {test_metric["regression"]["R2"]:.4f}')
        print('----------')
        
        # 更新日志
        log_info['train_total_loss'].append(train_total_loss)
        log_info['train_cls_loss'].append(train_cls_loss)
        log_info['train_reg_loss'].append(train_reg_loss)
        log_info['current_lambda'].append(current_lambda)
        log_info['current_lr'].append(current_lr)
        log_info['valid_metric'].append(val_metric)
        log_info['test_metric'].append(test_metric)
        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        # 当前验证PR-AUC（用于调度器、最佳模型和早停）
        cur_pr_auc = val_metric["classification"]["PR_AUC"]

        # 学习率调度（基于验证PR-AUC）
        post_step_lr = optimizer.param_groups[0]['lr']
        if ep >= args.warmup and ep >= args.step_start:
            prev_lr = current_lr
            if np.isfinite(cur_pr_auc):
                lr_sher.step(cur_pr_auc)
                post_step_lr = optimizer.param_groups[0]['lr']

                # 打印学习率状态
                if post_step_lr < prev_lr:
                    print(f'[学习率衰减] {prev_lr:.6f} → {post_step_lr:.6f}')
                elif np.isclose(post_step_lr, args.min_lr, atol=1e-12, rtol=0.0):
                    print(f'[当前学习率] 已达最小学习率 {post_step_lr:.6f}，停止衰减')
            else:
                print('[学习率调度] 当前验证PR-AUC为NaN，跳过本轮调度')

        # 保存最佳模型：以验证集PR-AUC为准（需提升超过auc_delta）
        if np.isfinite(cur_pr_auc):
            if cur_pr_auc > best_cls_pr_auc + args.auc_delta:
                best_cls_pr_auc = cur_pr_auc
                best_pr_auc_ep = ep + 1
                torch.save(model.state_dict(), best_model_dir)
                print(f'[最佳模型更新] 轮次: {best_pr_auc_ep}, 验证PR-AUC: {best_cls_pr_auc:.5f}')
        else:
            print('[最佳模型更新] 当前验证PR-AUC为NaN，跳过本轮AUC最优模型更新')

        # 真正早停：学习率到达最小值后，若PR-AUC连续 early_stop_patience 轮未提升则停止
        if early_stop_patience > 0:
            if post_step_lr <= args.min_lr + 1e-12:
                if np.isfinite(cur_pr_auc) and cur_pr_auc > min_lr_best_pr_auc + args.auc_delta:
                    min_lr_best_pr_auc = cur_pr_auc
                    min_lr_no_improve_epochs = 0
                    print(f'[早停计数] 最小学习率下PR-AUC提升至 {min_lr_best_pr_auc:.5f}，计数重置')
                else:
                    min_lr_no_improve_epochs += 1
                    if np.isfinite(cur_pr_auc):
                        print(
                            f'[早停计数] 最小学习率下PR-AUC未提升轮数: '
                            f'{min_lr_no_improve_epochs}/{early_stop_patience}'
                        )
                    else:
                        print(
                            f'[早停计数] 最小学习率下PR-AUC为NaN，按未提升计数: '
                            f'{min_lr_no_improve_epochs}/{early_stop_patience}'
                        )

                if min_lr_no_improve_epochs >= early_stop_patience:
                    early_stop_epoch = ep + 1
                    log_info['early_stop_epoch'] = early_stop_epoch
                    with open(log_dir, 'w') as Fout:
                        json.dump(log_info, Fout, indent=4)
                    print(
                        f'[早停触发] 最小学习率下PR-AUC连续 '
                        f'{early_stop_patience} 轮未提升，提前结束训练（第{early_stop_epoch}轮）'
                    )
                    break
            else:
                min_lr_no_improve_epochs = 0
                min_lr_best_pr_auc = -float('inf')

    # 输出最终结果
    print('\n[训练完成]')
    if early_stop_epoch is not None:
        print(f'[训练提前结束] 早停轮次: {early_stop_epoch}')
    if best_pr_auc_ep > 0:
        print(f'最佳模型：轮次 {best_pr_auc_ep}, 验证PR-AUC: {best_cls_pr_auc:.5f}, '
            f'验证F1 {log_info["valid_metric"][best_pr_auc_ep - 1]["classification"]["F1"]:.4f}, '
            f'验证R2 {log_info["valid_metric"][best_pr_auc_ep - 1]["regression"]["R2"]:.4f}, '
            f'验证MAE {log_info["valid_metric"][best_pr_auc_ep - 1]["regression"]["MAE"]:.4f}, '
            f'验证MSE {log_info["valid_metric"][best_pr_auc_ep - 1]["regression"]["MSE"]:.4f}')
    else:
        print('未产生可用的AUC最优模型（可能因验证集单类导致PR-AUC始终为NaN）')
    
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
