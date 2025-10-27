import torch
import numpy as np

from tqdm import tqdm
from torch.nn.functional import kl_div, mse_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    loader, model, optimizer, device, lambda_reg=0.5,  # lambda为回归损失权重
    total_heads=None, local_heads=0, warmup=False, has_reag=True   # 适配无条件场景
):
    model.train()
    total_loss = []
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
        cls_loss = torch.nn.functional.cross_entropy(cls_out, cls_label)  # 分类损失
        
        # --------------------------
        # 回归损失：仅对非NaN标签的样本计算
        # --------------------------
        # 筛选有效回归样本（非NaN）
        reg_valid_mask = torch.isfinite(reg_label)  # 有效样本为True，NaN为False
        num_valid_reg = reg_valid_mask.sum().item()
        
        if num_valid_reg > 0:
            # 只对有效样本计算MSE
            reg_pred_valid = reg_out.squeeze()[reg_valid_mask]
            reg_label_valid = reg_label[reg_valid_mask]
            reg_loss = torch.nn.functional.mse_loss(reg_pred_valid, reg_label_valid)
        else:
            # 无有效回归样本时，回归损失为0（不影响总损失）
            reg_loss = torch.tensor(0.0, device=device)

        # --------------------------
        # 总损失：分类损失 + λ×回归损失
        # --------------------------
        loss = cls_loss + lambda_reg * reg_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss.append(loss.item())
        
        if warmup:
            warmup_sher.step()
    
    return np.mean(total_loss)


def eval_joint(
    loader, model, device, total_heads=None, local_heads=0, return_raw=False, has_reag=False
):
    model.eval()
    # 分类任务：所有样本均参与（含无回归标签的样本）
    cls_true, cls_pred = [], []
    # 回归任务：仅保留非NaN标签的样本
    reg_true, reg_pred = [], []
    
    for batch_data in tqdm(loader):
        # 解析批次数据
        if has_reag:
            reac, prod, reag, cls_label, reg_label = batch_data
            reag = reag.to(device)
        else:
            reac, prod, cls_label, reg_label = batch_data
            reag = None
        
        reac, prod = reac.to(device), prod.to(device)
        
        # 生成掩码
        if local_heads > 0:
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None
        
        # 前向传播（获取双任务输出）
        with torch.no_grad():
            cls_out, reg_out = model(reac, prod, reag, cross_mask=cross_mask)
        
        # --------------------------
        # 分类结果处理（所有样本）
        # --------------------------
        cls_pred_batch = cls_out.argmax(dim=1).cpu().numpy()  # 取概率最大的类别
        cls_true_batch = cls_label.numpy()
        cls_true.append(cls_true_batch)
        cls_pred.append(cls_pred_batch)
        
        # --------------------------
        # 回归结果处理（仅非NaN样本）
        # --------------------------
        reg_pred_batch = torch.clamp(reg_out, 0).squeeze().cpu().numpy()  # 能垒非负，截断 < 0 的值为 0
        # reg_pred_batch = reg_out.squeeze().cpu().numpy()  # 不进行截断
        reg_label_batch = reg_label.numpy()  # 真实标签（可能含NaN）
        
        # 筛选有效样本（非NaN）
        reg_valid_mask = np.isfinite(reg_label_batch)
        if np.any(reg_valid_mask):
            reg_true.append(reg_label_batch[reg_valid_mask])
            reg_pred.append(reg_pred_batch[reg_valid_mask])
    
    # --------------------------
    # 计算分类指标（所有样本）
    # --------------------------
    cls_true = np.concatenate(cls_true, axis=0)
    cls_pred = np.concatenate(cls_pred, axis=0)
    cls_acc = float(np.mean(cls_true == cls_pred))
    
    # --------------------------
    # 计算回归指标（仅有效样本）
    # --------------------------
    if len(reg_true) == 0:
        # 无有效回归样本时，指标设为NaN
        reg_mae = reg_mse = reg_r2 = float('nan')
    else:
        reg_true = np.concatenate(reg_true, axis=0)
        reg_pred = np.concatenate(reg_pred, axis=0)
        reg_mae = float(mean_absolute_error(reg_true, reg_pred))
        reg_mse = float(mean_squared_error(reg_true, reg_pred))
        reg_r2 = float(r2_score(reg_true, reg_pred))
    
    # 整理结果
    result = {
        'classification': {'ACC': cls_acc},
        'regression': {'MAE': reg_mae, 'MSE': reg_mse, 'R2': reg_r2}
    }
    
    if return_raw:
        result['raw'] = {
            'cls_true': cls_true.tolist(), 
            'cls_pred': cls_pred.tolist(),
            'reg_true': reg_true.tolist() if len(reg_true) > 0 else [],
            'reg_pred': reg_pred.tolist() if len(reg_pred) > 0 else []
        }
    return result