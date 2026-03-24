import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.nn.functional import mse_loss, softmax, cross_entropy
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)

from ..tensor_utils import generate_local_global_mask


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_joint(
    loader, model, optimizer, device, lambda_reg=0.005,
    total_heads=None, local_heads=0, warmup=False,
    warmup_scheduler=None, warmup_total_steps=0, focal_alpha=0.5
):
    model.train()
    total_losses = []
    cls_losses = []
    reg_losses = []
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=focal_alpha)
    local_warmup_scheduler = warmup_scheduler
    local_warmup_total_steps = warmup_total_steps
    # Backward-compatible fallback: if caller only passes warmup=True.
    if local_warmup_scheduler is None and warmup:
        local_warmup_total_steps = max(len(loader) - 1, 1)
        local_warmup_scheduler = warmup_lr_scheduler(
            optimizer, local_warmup_total_steps, 5e-2
        )

    for reac, prod, cls_label, reg_label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
        cls_label = cls_label.to(device)
        reg_label = reg_label.to(device)

        if local_heads > 0:
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
        else:
            cross_mask = None

        cls_out, reg_out = model(reac, prod, None, cross_mask=cross_mask)
        cls_loss = focal_loss_fn(cls_out, cls_label)

        reg_valid_mask = torch.isfinite(reg_label)
        if reg_valid_mask.any():
            reg_pred_valid = reg_out.view(-1)[reg_valid_mask]
            reg_label_valid = reg_label[reg_valid_mask]
            reg_loss = mse_loss(reg_pred_valid, reg_label_valid)
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total_loss = cls_loss + lambda_reg * reg_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_losses.append(total_loss.item())
        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())
        if local_warmup_scheduler is not None:
            # Execute warmup only once globally. After warmup is done, stop stepping
            # this scheduler to avoid overriding later ReduceLROnPlateau updates.
            if (local_warmup_scheduler.last_epoch + 1) < local_warmup_total_steps:
                local_warmup_scheduler.step()

    return (
        np.mean(total_losses),
        np.mean(cls_losses),
        np.mean(reg_losses)
    )


def eval_joint(
    loader, model, device, total_heads=None, local_heads=0, return_raw=False,
    pos_label=1, lambda_reg=0.005, focal_alpha=0.5
):
    model.eval()
    cls_true, cls_pred, cls_scores = [], [], []
    reg_true, reg_pred = [], []
    val_total_loss = []
    val_cls_loss = []
    val_reg_loss = []
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=focal_alpha)

    for reac, prod, cls_label, reg_label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
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
            cls_loss = focal_loss_fn(cls_out, cls_label)

            reg_valid_mask = torch.isfinite(reg_label)
            reg_out_flat = reg_out.view(-1)
            if reg_valid_mask.any():
                reg_pred_valid = reg_out_flat[reg_valid_mask]
                reg_label_valid = reg_label[reg_valid_mask]
                reg_loss = mse_loss(reg_pred_valid, reg_label_valid)
            else:
                reg_loss = torch.tensor(0.0, device=device)

            total_loss = cls_loss + lambda_reg * reg_loss
            val_total_loss.append(total_loss.item())
            val_cls_loss.append(cls_loss.item())
            val_reg_loss.append(reg_loss.item())

            cls_pred_batch = cls_out.argmax(dim=1).cpu().numpy()
            cls_scores_batch = softmax(cls_out, dim=1)[:, pos_label].cpu().numpy()
            cls_true_batch = cls_label.cpu().numpy()

            reg_pred_batch = reg_out.view(-1).cpu().numpy()
            reg_label_batch = reg_label.cpu().numpy()
            reg_nan_mask = ~np.isfinite(reg_label_batch)
            reg_pred_batch[reg_nan_mask] = np.nan

            cls_true.append(cls_true_batch)
            cls_pred.append(cls_pred_batch)
            cls_scores.append(cls_scores_batch)
            reg_true.append(reg_label_batch)
            reg_pred.append(reg_pred_batch)

    val_total_loss_avg = np.mean(val_total_loss) if val_total_loss else 0.0
    val_cls_loss_avg = np.mean(val_cls_loss) if val_cls_loss else 0.0
    val_reg_loss_avg = np.mean(val_reg_loss) if val_reg_loss else 0.0

    cls_true = np.concatenate(cls_true, axis=0) if cls_true else np.array([])
    cls_pred = np.concatenate(cls_pred, axis=0) if cls_pred else np.array([])
    cls_scores = np.concatenate(cls_scores, axis=0) if cls_scores else np.array([])
    reg_true = np.concatenate(reg_true, axis=0) if reg_true else np.array([])
    reg_pred = np.concatenate(reg_pred, axis=0) if reg_pred else np.array([])

    cls_acc = float(np.mean(cls_true == cls_pred))
    cls_cm = confusion_matrix(cls_true, cls_pred)
    average_mode = 'binary'

    try:
        cls_precision = precision_score(
            cls_true, cls_pred, average=average_mode,
            pos_label=pos_label, zero_division=0
        )
        cls_recall = recall_score(
            cls_true, cls_pred, average=average_mode,
            pos_label=pos_label, zero_division=0
        )
        cls_f1 = f1_score(
            cls_true, cls_pred, average=average_mode,
            pos_label=pos_label, zero_division=0
        )
    except Exception as e:
        cls_precision = cls_recall = cls_f1 = 0.0
        print(f"[警告] 分类指标计算异常：{e}，指标设为0.0")

    cls_pr_auc = float('nan')
    cls_pr_best_f1 = float('nan')
    cls_pr_best_threshold = float('nan')
    cls_true_bin = (cls_true == pos_label).astype(np.int32)
    if np.unique(cls_true_bin).size >= 2:
        try:
            cls_pr_auc = float(average_precision_score(cls_true_bin, cls_scores))
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

    reg_valid_mask = np.isfinite(reg_true)
    if np.sum(reg_valid_mask) == 0:
        reg_mae = reg_mse = reg_r2 = float('nan')
    else:
        reg_true_valid = reg_true[reg_valid_mask]
        reg_pred_valid = reg_pred[reg_valid_mask]
        reg_mae = float(mean_absolute_error(reg_true_valid, reg_pred_valid))
        reg_mse = float(mean_squared_error(reg_true_valid, reg_pred_valid))
        reg_r2 = float(r2_score(reg_true_valid, reg_pred_valid))

    result = {
        'classification': {
            'ACC': cls_acc,
            'Precision': float(cls_precision),
            'Recall': float(cls_recall),
            'F1': float(cls_f1),
            'Confusion_Matrix': cls_cm.tolist(),
            'PR_AUC': cls_pr_auc,
            'PR_BEST_F1': cls_pr_best_f1,
            'PR_BEST_F1_THRESHOLD': cls_pr_best_threshold
        },
        'regression': {
            'MAE': reg_mae,
            'MSE': reg_mse,
            'R2': reg_r2
        },
        'validation_loss': {
            'total_loss': float(val_total_loss_avg),
            'cls_loss': float(val_cls_loss_avg),
            'reg_loss': float(val_reg_loss_avg)
        }
    }

    if return_raw:
        result['raw'] = {
            'cls_true': cls_true.tolist(),
            'cls_pred': cls_pred.tolist(),
            'cls_scores': cls_scores.tolist(),
            'reg_true': reg_true.tolist(),
            'reg_pred': reg_pred.tolist()
        }
    return result


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, scale=10.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.scale = scale
        self.class_weights = torch.tensor([1 - alpha, alpha], dtype=torch.float32)

    def forward(self, logits, labels):
        probs = softmax(logits, dim=1)
        p_t = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = cross_entropy(logits, labels, reduction="none")
        class_weights = self.class_weights.to(logits.device)
        alpha_t = class_weights[labels]
        focal_loss = (alpha_t * focal_weight * ce_loss).mean()
        return focal_loss * self.scale
