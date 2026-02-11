import argparse
import glob
import json
import pickle
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_utils import fix_seed
from utils.Dataset import RAlignDatasetBase, graph_col_fn
from model import JointModel, RAlignEncoder, build_cn_condition_encoder_with_eval

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def tie_reac_prod_params(encoder):
    for layer in encoder.layers:
        layer.prod_mpnn = layer.reac_mpnn


class SimpleRxnDataset(RAlignDatasetBase):
    """仅包含反应 SMILES 的推理数据集，返回 reac/prod 图和原始 SMILES。"""
    def __getitem__(self, idx):
        reac_mol, prod_mol = self.get_aligned_graphs(idx)
        return reac_mol, prod_mol, self.reactions[idx]


def simple_collate(batch):
    reacs, prods, raws = [], [], []
    for reac, prod, raw in batch:
        reacs.append(reac)
        prods.append(prod)
        raws.append(raw)
    return graph_col_fn(reacs), graph_col_fn(prods), raws


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
        update_last_edge=False,
        use_lg_lin=args.use_lg_lin
    )
    if args.share_reac_prod_encoder:
        tie_reac_prod_params(encoder)

    return JointModel(
        encoder=encoder,
        condition_encoder=condition_encoder,
        dim=args.dim,
        dropout=dropout,
        heads=args.heads,
        cls_out_dim=args.cls_out_dim
    )


def collect_model_paths(model_paths: List[str], main_dir: str) -> List[str]:
    if model_paths:
        return model_paths
    if main_dir:
        logs_dir = os.path.join(main_dir, "logs") if os.path.isdir(os.path.join(main_dir, "logs")) else main_dir
        paths = glob.glob(os.path.join(logs_dir, "**", "best_loss.pth"), recursive=True)
        if not paths:
            paths = glob.glob(os.path.join(logs_dir, "**", "best_model.pt"), recursive=True)
        return sorted(paths)
    raise ValueError("未找到模型，请提供 --model_paths 或 --main_dir")


def binary_entropy(p: float) -> float:
    """二分类熵，输入为正类概率。"""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def patch_numpy_for_pickle():
    """兼容新版本 numpy 序列化产生的 numpy._core.* 路径。"""
    import numpy as np
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)


def load_reactions(input_path: str) -> Tuple[List[str], Optional[List[dict]]]:
    """读取输入文件，返回反应 SMILES 列表和（可选的）PKL 原始数据。"""
    reactions: List[str] = []
    raw_pkl: Optional[List[dict]] = None
    if input_path.endswith(".pkl"):
        patch_numpy_for_pickle()
        with open(input_path, "rb") as fin:
            raw_pkl = pickle.load(fin)
        if not isinstance(raw_pkl, list):
            raise ValueError("PKL 文件格式需为包含字典的列表")
        for idx, item in enumerate(raw_pkl):
            if not isinstance(item, dict) or 'reaction_smiles_mapped' not in item:
                raise ValueError(f"PKL 第 {idx} 个元素缺少 reaction_smiles_mapped")
            reactions.append(str(item['reaction_smiles_mapped']))
    elif input_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(input_path)
        if 'Reaction' not in df.columns:
            raise ValueError("CSV 文件需包含 'Reaction' 列")
        reactions = df['Reaction'].astype(str).tolist()
    else:
        with open(input_path) as f:
            reactions = [ln.strip() for ln in f if ln.strip()]
    if not reactions:
        raise ValueError("未读取到任何反应 SMILES")
    return reactions, raw_pkl


def update_pkl_predictions(
    raw_pkl: List[dict],
    per_model_cls: np.ndarray,
    per_model_reg: np.ndarray
) -> List[dict]:
    """将预测结果与不确定性附加到 PKL 数据中。"""
    n_models, n_samples = per_model_cls.shape
    if len(raw_pkl) != n_samples:
        raise ValueError(f"PKL 样本数 {len(raw_pkl)} 与预测结果 {n_samples} 不一致")

    for idx, item in enumerate(raw_pkl):
        # 确保存在 reaction_prediction 列表
        if 'reaction_prediction' not in item or item['reaction_prediction'] is None:
            item['reaction_prediction'] = []

        cls_vals = per_model_cls[:, idx]
        reg_vals = per_model_reg[:, idx]

        mean_cls_prob = float(np.nanmean(cls_vals))
        cls_entropy = binary_entropy(mean_cls_prob)
        cls_entropy_rounded = float(np.round(cls_entropy, 3))
        reg_std = float(np.nanstd(reg_vals))
        reg_std_rounded = float(np.round(reg_std, 2))

        pred_entry = {
            "model_cls_prob": [float(x) for x in cls_vals.tolist()],
            "model_barrier": [float(x) for x in reg_vals.tolist()],
            "uncert_cls_entropy_rounded": cls_entropy_rounded,
            "uncert_reg_std": reg_std_rounded
        }
        item['reaction_prediction'].append(pred_entry)
    return raw_pkl


def main():
    parser = argparse.ArgumentParser("无标签反应列表的投票预测")
    parser.add_argument('--input', required=True, help='反应 SMILES 列表（csv含Reaction列、txt逐行或包含 reaction_smiles_mapped 的pkl）')
    parser.add_argument('--main_dir', type=str, default=None, help='训练输出根目录（默认 logs 下搜模型）')
    parser.add_argument('--model_paths', nargs='+', default=None, help='直接提供模型路径列表')
    parser.add_argument('--output', required=True, help='输出路径（CSV 或 PKL，取决于输入类型）')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--cls_out_dim', type=int, default=2)
    parser.add_argument('--local_heads', type=int, default=4)
    parser.add_argument('--use_lg_lin', action='store_true', help='启用reactant-only分支（需与训练一致）')
    parser.add_argument('--share_reac_prod_encoder', action='store_true', help='反应物/产物编码层共享参数（需与训练一致）')
    parser.add_argument('--use_condition', action='store_true')
    parser.add_argument('--condition_config', type=str, default='')
    parser.add_argument('--condition_both', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.use_condition and not args.condition_config:
        raise ValueError("启用条件编码器需提供 --condition_config")

    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}') if (torch.cuda.is_available() and args.device >= 0) else torch.device('cpu')

    paths = collect_model_paths(args.model_paths, args.main_dir)
    if len(paths) == 0:
        raise ValueError("未找到模型文件")
    print(f"[INFO] 模型数: {len(paths)}")

    reactions, raw_pkl = load_reactions(args.input)
    dataset = SimpleRxnDataset(reactions)
    loader = DataLoader(
        dataset, batch_size=args.bs, shuffle=False,
        collate_fn=simple_collate, num_workers=args.num_worker, pin_memory=True
    )

    # 保存所有模型预测（分类用正类概率，回归用数值）
    per_model_cls = []
    per_model_reg = []

    for p in paths:
        model = build_model(args, dropout=0.0).to(device)
        state = torch.load(p, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"[INFO] 加载模型: {p}")

        cls_list, reg_list = [], []
        from tqdm import tqdm
        with torch.no_grad():
            for reac, prod, raws in tqdm(loader, desc=f"Infer {os.path.basename(p)}"):
                reac = reac.to(device)
                prod = prod.to(device)
                cls_out, reg_out = model(reac, prod, None, cross_mask=None)
                cls_prob = torch.softmax(cls_out, dim=1)[:, 1].cpu().numpy()
                reg_pred = reg_out.view(-1).cpu().numpy()
                cls_list.append(cls_prob)
                reg_list.append(reg_pred)

        cls_concat = np.concatenate(cls_list, axis=0)
        reg_concat = np.concatenate(reg_list, axis=0)
        per_model_cls.append(cls_concat)
        per_model_reg.append(reg_concat)

    per_model_cls = np.stack(per_model_cls, axis=0)
    per_model_reg = np.stack(per_model_reg, axis=0)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if raw_pkl is not None:
        # PKL 输入：将预测结果写回 PKL
        updated_pkl = update_pkl_predictions(raw_pkl, per_model_cls, per_model_reg)
        with open(args.output, "wb") as fout:
            pickle.dump(updated_pkl, fout)
        print(f"[INFO] PKL 预测完成，保存至 {args.output}")
    else:
        # 汇总到 DataFrame
        import pandas as pd
        data = {'Reaction': reactions}
        for i in range(len(paths)):
            data[f'model{i+1}_cls_prob'] = per_model_cls[i]
            data[f'model{i+1}_barrier'] = per_model_reg[i]
        df_out = pd.DataFrame(data)
        df_out.to_csv(args.output, index=False)
        print(f"[INFO] 预测完成，保存至 {args.output}")


if __name__ == '__main__':
    main()
