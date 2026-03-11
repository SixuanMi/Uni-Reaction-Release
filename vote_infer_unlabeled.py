import argparse
import glob
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_utils import fix_seed
from utils.model_factory import build_joint_model, resolve_device
from utils.Dataset import RAlignDatasetBase, graph_col_fn

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


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


def collect_model_paths(model_paths: List[str], main_dir: Optional[str]) -> List[str]:
    if model_paths:
        return model_paths
    if main_dir:
        logs_dir = os.path.join(main_dir, "logs") if os.path.isdir(os.path.join(main_dir, "logs")) else main_dir
        paths = glob.glob(os.path.join(logs_dir, "**", "best_model.pth"), recursive=True)
        if not paths:
            paths = glob.glob(os.path.join(logs_dir, "**", "best_model.pt"), recursive=True)
        return sorted(paths)
    raise ValueError("未找到模型，请提供 --model_paths 或 --main_dir")


def parse_device_ids(devices: Optional[str], fallback_device: int) -> List[int]:
    if devices is None:
        return [fallback_device]
    ids = []
    for token in devices.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError as e:
            raise ValueError(f"--devices 中存在非法设备编号: {token}") from e
    if not ids:
        raise ValueError("--devices 不能为空，请提供如 0,1,2 的设备列表")
    return ids


def resolve_reaction_column(df: pd.DataFrame, reaction_col: str) -> str:
    if reaction_col not in df.columns:
        raise ValueError(
            f"--reaction_col={reaction_col} 不存在于输入 CSV 中。"
            f"当前列为: {list(df.columns)}"
        )
    return reaction_col


def normalize_state_dict(raw_state):
    state = raw_state
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint 内容不是可用的 state_dict")

    # 兼容 DataParallel 保存的 "module." 前缀
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def build_model_from_checkpoint(
    checkpoint_path: str,
    model_cfg: Dict,
    fusion_mode: str,
    device: torch.device
):
    raw_state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = normalize_state_dict(raw_state)

    build_cfg = dict(model_cfg)
    build_cfg["fusion_mode"] = fusion_mode
    model_args = SimpleNamespace(**build_cfg)
    model = build_joint_model(model_args, dropout=0.0).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"加载模型失败: {checkpoint_path}。请确认 --fusion_mode 与训练一致。"
        ) from e
    model.eval()
    return model


def infer_models_on_device(
    device_id: int,
    model_paths: List[str],
    input_path: str,
    reaction_col: str,
    model_cfg: Dict,
    bs: int,
    num_worker: int,
    fusion_mode: str,
    seed: int
) -> Dict:
    fix_seed(seed)
    device = resolve_device(device_id)

    df_rxn = pd.read_csv(input_path, usecols=[reaction_col])
    reactions = df_rxn[reaction_col].astype(str).tolist()
    dataset = SimpleRxnDataset(reactions)
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=simple_collate,
        num_workers=num_worker,
        pin_memory=(device.type == "cuda")
    )

    outputs = []
    for idx, p in enumerate(model_paths, start=1):
        model = build_model_from_checkpoint(
            checkpoint_path=p,
            model_cfg=model_cfg,
            fusion_mode=fusion_mode,
            device=device
        )
        cls_list, reg_list = [], []
        with torch.no_grad():
            for reac, prod, _ in tqdm(
                loader,
                desc=f"Infer {os.path.basename(p)} @ cuda:{device_id}" if device.type == "cuda" else f"Infer {os.path.basename(p)} @ cpu",
                leave=False
            ):
                reac = reac.to(device)
                prod = prod.to(device)
                cls_out, reg_out = model(reac, prod, None, cross_mask=None)
                cls_prob = torch.softmax(cls_out, dim=1)[:, 1].cpu().numpy()
                reg_pred = reg_out.view(-1).cpu().numpy()
                cls_list.append(cls_prob)
                reg_list.append(reg_pred)

        cls_concat = np.concatenate(cls_list, axis=0).astype(np.float32, copy=False)
        reg_concat = np.concatenate(reg_list, axis=0).astype(np.float32, copy=False)
        outputs.append({
            "model_path": p,
            "cls_prob": cls_concat,
            "reg_pred": reg_concat
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[INFO] 设备 {device} 完成模型 ({idx}/{len(model_paths)}): "
            f"{p} (fusion_mode={fusion_mode})"
        )
    return {"device": str(device), "outputs": outputs}


def main():
    parser = argparse.ArgumentParser("无标签反应列表的投票预测（CSV 输入输出）")
    parser.add_argument("--input", required=True, help="输入 CSV 路径")
    parser.add_argument("--reaction_col", type=str, default="stereo_rsmi", help="反应 SMILES 列名")
    parser.add_argument("--main_dir", type=str, default=None, help="训练输出根目录（默认 logs 下搜模型）")
    parser.add_argument("--model_paths", nargs="+", default=None, help="直接提供模型路径列表")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=5)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--devices", type=str, default=None, help="逗号分隔设备ID，如 0,1,2；用于多卡并行推理")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--local_heads", type=int, default=4)
    parser.add_argument(
        "--fusion_mode", type=str, default="film", choices=["legacy", "film"],
        help="R/P融合方式（需与训练一致）"
    )
    args = parser.parse_args()
    print(args)

    if not args.input.endswith(".csv"):
        raise ValueError("vote_infer_unlabeled 仅支持 CSV 输入")
    if not args.output.endswith(".csv"):
        raise ValueError("vote_infer_unlabeled 仅支持 CSV 输出")

    fix_seed(args.seed)
    device_ids = parse_device_ids(args.devices, args.device)
    print(f"[INFO] 推理设备: {device_ids}")

    df_in = pd.read_csv(args.input)
    reaction_col = resolve_reaction_column(df_in, args.reaction_col)
    if df_in[reaction_col].isna().any():
        raise ValueError(f"反应列 {reaction_col} 存在空值，请先清理")
    reactions = df_in[reaction_col].astype(str).tolist()
    if not reactions:
        raise ValueError("未读取到任何反应 SMILES")
    print(f"[INFO] 读取样本数: {len(reactions)}, 反应列: {reaction_col}")

    paths = collect_model_paths(args.model_paths, args.main_dir)
    if len(paths) == 0:
        raise ValueError("未找到模型文件")
    print(f"[INFO] 模型数: {len(paths)}")

    model_cfg = {
        "dim": args.dim,
        "heads": args.heads,
        "n_layer": args.n_layer,
        "negative_slope": args.negative_slope,
    }

    # 将模型按设备轮询分配，保证多卡负载均衡
    assigned = {d: [] for d in device_ids}
    for idx, p in enumerate(paths):
        assigned_device = device_ids[idx % len(device_ids)]
        assigned[assigned_device].append(p)

    device_tasks = [(d, assigned[d]) for d in device_ids if assigned[d]]
    per_model_outputs = [None] * len(paths)
    path_to_idx = {p: i for i, p in enumerate(paths)}

    if len(device_tasks) == 1:
        d, model_paths = device_tasks[0]
        print(f"[INFO] 单卡推理模式: device={d}, 模型数={len(model_paths)}")
        result = infer_models_on_device(
            device_id=d,
            model_paths=model_paths,
            input_path=args.input,
            reaction_col=reaction_col,
            model_cfg=model_cfg,
            bs=args.bs,
            num_worker=args.num_worker,
            fusion_mode=args.fusion_mode,
            seed=args.seed
        )
        for out in result["outputs"]:
            per_model_outputs[path_to_idx[out["model_path"]]] = out
    else:
        max_parallel = len(device_tasks)
        per_proc_num_worker = args.num_worker // max_parallel if args.num_worker > 0 else 0
        print(
            f"[INFO] 多卡并行推理模式: 并行进程={max_parallel}, "
            f"每进程DataLoader workers={per_proc_num_worker}"
        )
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_parallel, mp_context=ctx) as executor:
            futures = {}
            for d, model_paths in device_tasks:
                fut = executor.submit(
                    infer_models_on_device,
                    d,
                    model_paths,
                    args.input,
                    reaction_col,
                    model_cfg,
                    args.bs,
                    per_proc_num_worker,
                    args.fusion_mode,
                    args.seed
                )
                futures[fut] = (d, model_paths)

            finished = 0
            for fut in as_completed(futures):
                d, model_paths = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    raise RuntimeError(
                        f"设备 {d} 推理失败（负责模型数={len(model_paths)}）"
                    ) from e
                finished += 1
                print(f"[INFO] 设备任务完成 ({finished}/{len(device_tasks)}): {result['device']}")
                for out in result["outputs"]:
                    per_model_outputs[path_to_idx[out["model_path"]]] = out

    if any(x is None for x in per_model_outputs):
        raise RuntimeError("存在模型未完成推理，无法汇总输出")

    per_model_cls = np.stack([x["cls_prob"] for x in per_model_outputs], axis=0)
    per_model_reg = np.stack([x["reg_pred"] for x in per_model_outputs], axis=0)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 保留输入 CSV 的全部原始列，便于后续 active_select 与回溯
    df_out = df_in.copy()
    if "Reaction" not in df_out.columns:
        df_out["Reaction"] = reactions
    for i in range(len(paths)):
        df_out[f"model{i+1}_cls_prob"] = per_model_cls[i]
        df_out[f"model{i+1}_barrier"] = per_model_reg[i]
    df_out.to_csv(args.output, index=False)

    print(f"[INFO] 预测完成，保存至 {args.output}")
    print(f"[INFO] 使用 fusion_mode: {args.fusion_mode}")


if __name__ == "__main__":
    main()
