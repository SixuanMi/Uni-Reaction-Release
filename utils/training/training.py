import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.nn.functional import kl_div, mse_loss, softmax, cross_entropy
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)

from ..tensor_utils import (
    generate_local_global_mask, generate_tgt_mask, calc_trans_loss,
    correct_trans_output, data_eval_trans, convert_log_into_label
)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def eval_mol_yield(
    loader, model, device, total_heads=None, local_heads=0, return_raw=False
):
    model, ytrue, ypred = model.eval(), [], []
    for reac, prod, reag, label in tqdm(loader):
        reac, prod, reag = reac.to(device), prod.to(device), reag.to(device)
        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None

        with torch.no_grad():
            res = model(reac, prod, reag, cross_mask=cross_mask)
            if res.shape[-1] == 2:
                res = res.softmax(dim=-1)[:, 0] * 100
                ytrue.append(label.numpy())
                ypred.append(res.cpu().numpy())
            else:
                res = torch.clamp(res, 0, 1) * 100
                ytrue.append(label.numpy())
                ypred.append(res.cpu().numpy())

    ypred = np.concatenate(ypred, axis=0)
    ytrue = np.concatenate(ytrue, axis=0)

    result = {
        'MAE': float(mean_absolute_error(ytrue, ypred)),
        'MSE': float(mean_squared_error(ytrue, ypred)),
        'R2': float(r2_score(ytrue, ypred))
    }

    if return_raw:
        result['label'] = ytrue.flatten().tolist()
        result['prediction'] = ypred.flatten().tolist()
    return result


def train_regression(
    loader, model, optimizer, device, total_heads=None,
    local_heads=0, warmup=False, has_reag=True
):
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for batch_data in tqdm(loader):
        if has_reag:
            reac, prod, reag, label = batch_data
            reag = reag.to(device)
        else:
            reac, prod, label = batch_data
            reag = None
        reac, prod, label = reac.to(device), prod.to(device), label.to(device)
        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None

        res = model(reac, prod, reag, cross_mask=cross_mask)
        assert res.shape[-1] == 1, 'requires single output'
        loss = mse_loss(label, res.squeeze(dim=-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_regression(
    loader, model, device, total_heads=None, local_heads=0,
    return_raw=False, has_reag=True,
):
    model, ytrue, ypred = model.eval(), [], []
    for batch_data in tqdm(loader):
        if has_reag:
            reac, prod, reag, label = batch_data
            reag = reag.to(device)
        else:
            reac, prod, label = batch_data
            reag = None
        reac, prod = reac.to(device), prod.to(device)
        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None

        with torch.no_grad():
            res = model(reac, prod, reag, cross_mask=cross_mask)
            ytrue.append(label.numpy())
            ypred.append(res.cpu().numpy())

    ypred = np.concatenate(ypred, axis=0)
    ytrue = np.concatenate(ytrue, axis=0)

    result = {
        'MAE': float(mean_absolute_error(ytrue, ypred)),
        'MSE': float(mean_squared_error(ytrue, ypred)),
        'R2': float(r2_score(ytrue, ypred))
    }

    if return_raw:
        result['label'] = ytrue.flatten().tolist()
        result['prediction'] = ypred.flatten().tolist()
    return result


def train_gen(
    loader, model, optimizer, device, pad_idx, toker,
    total_heads=None, local_heads=0, warmup=False,
):
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for reac, prod, label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
        tgt = toker.encode2d(label)
        tgt = torch.LongTensor(tgt).to(device)

        trans_dec_ip = tgt[:, :-1]
        trans_dec_op = tgt[:, 1:]
        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, pad_idx=pad_idx, device=device
        )

        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, trans_dec_ip.shape[1],
                total_heads, local_heads
            )
        else:
            cross_mask = None

        trans_logs = model(
            reac_graph=reac, prod_graph=prod, tgt=trans_dec_ip,
            tgt_mask=diag_mask, cross_mask=cross_mask,
            tgt_key_padding_mask=trans_op_mask
        )

        loss = calc_trans_loss(trans_logs, trans_dec_op, pad_idx)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_gen(
    loader, model, device, pad_idx, end_idx,
    toker, total_heads=None, local_heads=0
):
    model, accx = model.eval(), []
    for reac, prod, label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
        tgt = toker.encode2d(label)
        tgt = torch.LongTensor(tgt).to(device)

        trans_dec_ip = tgt[:, :-1]
        trans_dec_op = tgt[:, 1:]
        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, pad_idx=pad_idx, device=device
        )

        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, trans_dec_ip.shape[1],
                total_heads, local_heads
            )
        else:
            cross_mask = None

        with torch.no_grad():
            trans_logs = model(
                reac_graph=reac, prod_graph=prod, tgt=trans_dec_ip,
                tgt_mask=diag_mask, cross_mask=cross_mask,
                tgt_key_padding_mask=trans_op_mask
            )

        trans_pred = convert_log_into_label(trans_logs, mod='softmax')
        trans_pred = correct_trans_output(trans_pred, end_idx, pad_idx)
        trans_acc = data_eval_trans(trans_pred, trans_dec_op, True)
        accx.append(trans_acc)

    accx = torch.cat(accx, dim=0).float()
    return accx.mean().item()


def train_uspto_condition(
    loader, model, optimizer, device, total_heads=None,
    local_heads=0, warmup=False
):
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for reac, prod, label in tqdm(loader):
        reac, prod, label = reac.to(device), prod.to(device), label.to(device)
        tgt_in, tgt_out = label[:, :-1], label[:, 1:]

        pad_mask, sub_mask = generate_tgt_mask(tgt_in, -1000, device)

        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, tgt_in.shape[1], total_heads, local_heads
            )
        else:
            cross_mask = None

        res = model(
            reac, prod, tgt_in, tgt_mask=sub_mask,
            tgt_key_padding_mask=pad_mask, cross_mask=cross_mask
        )

        loss = calc_trans_loss(res, tgt_out, -1000)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_uspto_condition(
    loader, model, device, total_heads=None, local_heads=0
):
    model, accs, gt = model.eval(), [], []
    for reac, prod, label in tqdm(loader):
        reac, prod, label = reac.to(device), prod.to(device), label.to(device)
        tgt_in, tgt_out = label[:, :-1], label[:, 1:]
        pad_mask, sub_mask = generate_tgt_mask(tgt_in, -1000, device)

        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, tgt_in.shape[1], total_heads, local_heads
            )
        else:
            cross_mask = None

        with torch.no_grad():
            res = model(
                reac, prod, tgt_in, tgt_mask=sub_mask,
                tgt_key_padding_mask=pad_mask, cross_mask=cross_mask
            )

            result = convert_log_into_label(res, mod='softmax')

        accs.append(result)
        gt.append(tgt_out)

    accs = torch.cat(accs, dim=0)
    gt = torch.cat(gt, dim=0)

    keys = ['catalyst', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    results, overall = {}, None
    for idx, k in enumerate(keys):
        results[k] = accs[:, idx] == gt[:, idx]
        if idx == 0:
            overall = accs[:, idx] == gt[:, idx]
        else:
            overall &= (accs[:, idx] == gt[:, idx])

    results['overall'] = overall
    results = {k: v.float().mean().item() for k, v in results.items()}
    return results


def train_mol_yield_freeze(
    loader, model, optimizer, device, total_heads=None, local_heads=0,
    warmup=False, loss_fun='kl', freeze_layers=None
):
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    if freeze_layers is not None:
        for x in freeze_layers:
            x.eval()

    for reac, prod, reag, label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
        reag, label = reag.to(device), label.to(device)
        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None

        res = model(reac, prod, reag, cross_mask=cross_mask)
        if loss_fun == 'kl':
            assert res.shape[-1] == 2, 'kl requires two outputs'
            res = torch.log_softmax(res, dim=-1)
            sm_label = torch.stack([label, 100 - label], dim=1) / 100
            loss = kl_div(res, sm_label, reduction='batchmean')
        else:
            assert loss_fun == 'mse', f'Invalid loss_fun {loss_fun}'
            assert res.shape[-1] == 1, 'requires single output'
            loss = mse_loss(label / 100, res.squeeze(dim=-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def train_joint(
    loader, model, optimizer, device, lambda_reg=0.005,  # lambda为回归损失权重
    total_heads=None, local_heads=0, warmup=False, has_reag=True   # 适配无条件场景
):
    model.train()
    total_losses = []  # 总损失
    cls_losses = []    # 分类损失
    reg_losses = []    # 回归损失（未乘λ）
    # total_loss = []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)
    
    for batch_data in tqdm(loader):
        # 假设batch包含分类标签（cls_label）和回归标签（reg_label）
        if has_reag:
            reac, prod, reag, cls_label, reg_label = batch_data
            reag = reag.to(device)
        else:
            reac, prod, cls_label, reg_label = batch_data
            reag = None
        
        # 数据迁移到设备
        reac, prod = reac.to(device), prod.to(device)
        cls_label = cls_label.to(device)  # 分类标签（整数，必选）
        reg_label = reg_label.to(device)  # 回归标签（可能含NaN，float32）
        
        # 生成掩码（与原有逻辑一致）
        if local_heads > 0:
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None
        
        # 前向传播
        cls_out, reg_out = model(reac, prod, reag, cross_mask=cross_mask)
        
        # --------------------------
        # 分类损失：对所有样本计算
        # --------------------------
        # cls_loss = torch.nn.functional.cross_entropy(cls_out, cls_label)  # 分类损失
        cls_loss = FocalLoss(gamma=2.0)(cls_out, cls_label)  # 用Focal Loss替代，融入置信度权重
        
        # --------------------------
        # 回归损失：仅对非NaN标签的样本计算
        # --------------------------
        # 筛选有效回归样本（非NaN）
        reg_valid_mask = torch.isfinite(reg_label)  # 有效样本为True，NaN为False
        num_valid_reg = reg_valid_mask.sum().item()
        
        if num_valid_reg > 0:
            # 只对有效样本计算MSE
            # 用 view(-1) 避免 batch_size=1 时 squeeze 变成标量导致索引异常
            reg_pred_valid = reg_out.view(-1)[reg_valid_mask]
            reg_label_valid = reg_label[reg_valid_mask]
            reg_loss = mse_loss(reg_pred_valid, reg_label_valid)
        else:
            # 无有效回归样本时，回归损失为0（不影响总损失）
            reg_loss = torch.tensor(0.0, device=device)

        # --------------------------
        # 总损失：分类损失 + λ×回归损失
        # --------------------------
        total_loss = cls_loss + lambda_reg * reg_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录损失（转为标量）
        total_losses.append(total_loss.item())
        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())  # 这里记录的是未乘λ的原始回归损失
        
        if warmup:
            warmup_sher.step()
    
    # 返回平均总损失、平均分类损失、平均回归损失（便于打印）
    return (np.mean(total_losses), 
            np.mean(cls_losses), 
            np.mean(reg_losses))


def eval_joint(
    loader, model, device, total_heads=None, local_heads=0, return_raw=False, has_reag=False,
    num_classes=2,  # 新增：分类任务类别数（默认二分类，多分类需手动指定）
    pos_label=1,    # 新增：正类标签（默认1，即“实际为真”的标签值）
    lambda_reg=0.005  # lambda为回归损失权重
):
    model.eval()
    # 分类任务：新增 cls_scores 收集正类置信度得分
    cls_true, cls_pred, cls_scores = [], [], []  # 新增 cls_scores
    # 回归任务：原有逻辑完全不变
    reg_true, reg_pred = [], []
    # 新增：验证集损失统计（完全复用训练时的损失逻辑）
    val_total_loss = []  # 总损失（分类 + λ×回归）
    val_cls_loss = []    # 分类损失（FocalLoss）
    val_reg_loss = []    # 回归损失（有效样本MSE，未乘λ）
    
    for batch_data in tqdm(loader):
        # 解析批次数据（原有逻辑不变）
        if has_reag:
            reac, prod, reag, cls_label, reg_label = batch_data
            reag = reag.to(device)
        else:
            reac, prod, cls_label, reg_label = batch_data
            reag = None
        
        reac, prod = reac.to(device), prod.to(device)
        cls_label = cls_label.to(device)
        reg_label = reg_label.to(device)
            
        # 生成掩码（原有逻辑不变）
        if local_heads > 0:
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None
        
        # 前向传播（原有逻辑不变）
        with torch.no_grad():
            cls_out, reg_out = model(reac, prod, reag, cross_mask=cross_mask)

            # --------------------------
            # 损失计算（与train_joint完全一致）
            # --------------------------
            # 分类损失
            cls_loss = FocalLoss(gamma=2.0)(cls_out, cls_label)
            # 回归损失
            reg_valid_mask = torch.isfinite(reg_label)
            num_valid_reg = reg_valid_mask.sum().item()
            reg_out_flat = reg_out.view(-1)
            if num_valid_reg > 0:
                reg_pred_valid = reg_out_flat[reg_valid_mask]
                reg_label_valid = reg_label[reg_valid_mask]
                reg_loss = mse_loss(reg_pred_valid, reg_label_valid)
            else:
                reg_loss = torch.tensor(0.0, device=device)
            # 总损失
            total_loss = cls_loss + lambda_reg * reg_loss
            
            # 记录当前批次损失（与train逻辑一致）
            val_total_loss.append(total_loss.item())
            val_cls_loss.append(cls_loss.item())
            val_reg_loss.append(reg_loss.item())
        
            # --------------------------
            # 标签/得分收集（原有逻辑保留）
            # --------------------------
            # 分类结果：预测标签 + 正类得分
            cls_pred_batch = cls_out.argmax(dim=1).cpu().numpy()
            cls_scores_batch = softmax(cls_out, dim=1)[:, pos_label].cpu().numpy()
            cls_true_batch = cls_label.cpu().numpy()
            
            # 回归结果：保留NaN
            reg_pred_batch = torch.clamp(reg_out, 0).view(-1).cpu().numpy()
            reg_label_batch = reg_label.cpu().numpy()
            reg_nan_mask = ~np.isfinite(reg_label_batch)
            reg_pred_batch[reg_nan_mask] = np.nan
            
            # 累加数据
            cls_true.append(cls_true_batch)
            cls_pred.append(cls_pred_batch)
            cls_scores.append(cls_scores_batch)
            reg_true.append(reg_label_batch)
            reg_pred.append(reg_pred_batch)

    # --------------------------
    # 计算平均损失（与train完全一致：直接求列表均值）
    # --------------------------
    val_total_loss_avg = np.mean(val_total_loss) if val_total_loss else 0.0
    val_cls_loss_avg = np.mean(val_cls_loss) if val_cls_loss else 0.0
    val_reg_loss_avg = np.mean(val_reg_loss) if val_reg_loss else 0.0
    
    # --------------------------
    # 拼接所有批次结果（原有逻辑不变）
    # --------------------------
    cls_true = np.concatenate(cls_true, axis=0) if cls_true else np.array([])
    cls_pred = np.concatenate(cls_pred, axis=0) if cls_pred else np.array([])
    cls_scores = np.concatenate(cls_scores, axis=0) if cls_scores else np.array([])
    reg_true = np.concatenate(reg_true, axis=0) if reg_true else np.array([])
    reg_pred = np.concatenate(reg_pred, axis=0) if reg_pred else np.array([])
    
    # --------------------------
    # 新增：分类任务高级指标计算（核心部分）
    # --------------------------
    # 1. 基础准确率（原有指标保留）
    cls_acc = float(np.mean(cls_true == cls_pred))
    
    # 2. 混淆矩阵（支持二分类/多分类）
    cls_cm = confusion_matrix(cls_true, cls_pred)  # 形状：[num_classes, num_classes]
    
    # 3. 二分类/多分类指标适配（默认二分类，多分类自动加权）
    average_mode = 'binary' if num_classes == 2 else 'weighted'
    
    # 4. 核心指标计算（处理极端情况：某类无样本时返回0.0）
    try:
        # 精确率（Precision）
        cls_precision = precision_score(
            cls_true, cls_pred, average=average_mode, pos_label=pos_label, zero_division=0
        )
        # 召回率（Recall）= 1 - 漏检率
        cls_recall = recall_score(
            cls_true, cls_pred, average=average_mode, pos_label=pos_label, zero_division=0
        )
        # F1分数（综合Precision和Recall）
        cls_f1 = f1_score(
            cls_true, cls_pred, average=average_mode, pos_label=pos_label, zero_division=0
        )

    except Exception as e:
        # 极端情况（如所有样本为同一类）下指标设为0.0
        cls_precision = cls_recall = cls_f1 = 0.0
        print(f"[警告] 分类指标计算异常：{e}，指标设为0.0")

    # 5. PR-AUC（二分类主任务）
    # 注意：PR-AUC 需要同时存在正负样本，否则此处按不可定义处理（返回 NaN）
    cls_pr_auc = float('nan')
    cls_pr_best_f1 = float('nan')
    cls_pr_best_threshold = float('nan')
    if num_classes == 2:
        cls_true_bin = (cls_true == pos_label).astype(np.int32)
        if np.unique(cls_true_bin).size >= 2:
            try:
                cls_pr_auc = float(average_precision_score(cls_true_bin, cls_scores))
                # 从PR曲线中选取 max-F1 对应阈值作为可解释分类阈值
                pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
                    cls_true_bin, cls_scores
                )
                if pr_thresholds.size > 0:
                    pr_f1 = 2 * pr_precision[:-1] * pr_recall[:-1] / np.clip(
                        pr_precision[:-1] + pr_recall[:-1], 1e-12, None
                    )
                    best_idx = int(np.nanargmax(pr_f1))
                    cls_pr_best_f1 = float(pr_f1[best_idx])
                    cls_pr_best_threshold = float(pr_thresholds[best_idx])
            except Exception as e:
                print(f"[警告] PR-AUC 计算异常：{e}，指标设为NaN")
    
    # --------------------------
    # 回归指标计算（原有逻辑完全不变）
    # --------------------------
    reg_valid_mask = np.isfinite(reg_true)
    if np.sum(reg_valid_mask) == 0:
        reg_mae = reg_mse = reg_r2 = float('nan')
    else:
        reg_true_valid = reg_true[reg_valid_mask]
        reg_pred_valid = reg_pred[reg_valid_mask]
        reg_mae = float(mean_absolute_error(reg_true_valid, reg_pred_valid))
        reg_mse = float(mean_squared_error(reg_true_valid, reg_pred_valid))
        reg_r2 = float(r2_score(reg_true_valid, reg_pred_valid))
    
    # --------------------------
    # 整理结果（新增分类高级指标）
    # --------------------------
    result = {
        'classification': {
            'ACC': cls_acc,
            'Precision': cls_precision,  # 精确率
            'Recall': cls_recall,        # 召回率
            'F1': cls_f1,                # F1分数
            'PR_AUC': cls_pr_auc,        # PR曲线下面积（二分类）
            'PR_BEST_F1': cls_pr_best_f1,  # PR曲线中最大F1
            'PR_BEST_F1_THRESHOLD': cls_pr_best_threshold,  # 最大F1对应阈值
            'Confusion_Matrix': cls_cm.tolist()  # 混淆矩阵（转为列表方便日志存储）
        },
        'regression': {'MAE': reg_mae, 'MSE': reg_mse, 'R2': reg_r2},
        # 新增：验证集损失（与训练损失逻辑一致）
        'validation_loss': {
            'total_loss': val_total_loss_avg,
            'cls_loss': val_cls_loss_avg,
            'reg_loss': val_reg_loss_avg,
            'lambda_reg': lambda_reg  # 记录当前λ，便于日志分析
        }
    }
    
    if return_raw:
        # 原始数据（原有逻辑不变）
        result['raw'] = {
            'cls_true': cls_true.tolist(), 
            'cls_pred': cls_pred.tolist(),
            'cls_scores': cls_scores.tolist(),  # 新增：正类置信度得分
            'reg_true': reg_true.tolist(),
            'reg_pred': reg_pred.tolist()
        }
    return result


# 定义Focal Loss（二分类）
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, scale=10.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # 聚焦参数：抑制易分样本，默认2.0
        self.alpha = alpha  # 正类权重：解决类别不平衡，默认0.5（不调整）
        # 预定义二分类的类别权重（负类权重=1-alpha，正类权重=alpha）
        self.scale = scale  # 损失缩放系数
        self.class_weights = torch.tensor([1 - alpha, alpha], dtype=torch.float32)

    def forward(self, logits, labels):
        # logits: 模型输出，shape=(batch_size, 2)（未经过softmax）
        # labels: 真实标签，shape=(batch_size,)（类别索引0/1，0=负类，1=正类）
        
        # 1. 计算每个样本对应真实标签的概率（p_t）
        probs = softmax(logits, dim=1)  # (batch_size, 2)：每个类别的概率
        # 提取真实标签对应的概率（gather避免维度冲突）
        p_t = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)  # (batch_size,)
        
        # 2. 计算Focal权重：(1-p_t)^gamma（易分样本权重小，难分样本权重大）
        focal_weight = (1 - p_t) ** self.gamma  # (batch_size,)
        
        # 3. 计算基础交叉熵损失（保留每个样本的损失）
        ce_loss = cross_entropy(logits, labels, reduction="none")  # (batch_size,)
        
        # 4. 加入类别平衡权重（自动适配设备，避免GPU/CPU冲突）
        class_weights = self.class_weights.to(logits.device)
        alpha_t = class_weights[labels]  # (batch_size,)：每个样本对应类别的权重
        
        # 5. 最终Focal Loss（平均到每个样本）
        focal_loss = (alpha_t * focal_weight * ce_loss).mean()
        return focal_loss * self.scale
