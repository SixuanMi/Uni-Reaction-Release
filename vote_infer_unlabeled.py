import argparse
import glob
import multiprocessing as mp
import os
import queue
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


def infer_loaded_models_on_reactions(
    models: List,
    model_paths: List[str],
    reactions: List[str],
    device: torch.device,
    device_id: int,
    bs: int,
    num_worker: int,
) -> List[Dict]:
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
    for model, p in zip(models, model_paths):
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

        if cls_list:
            cls_concat = np.concatenate(cls_list, axis=0).astype(np.float32, copy=False)
            reg_concat = np.concatenate(reg_list, axis=0).astype(np.float32, copy=False)
        else:
            cls_concat = np.array([], dtype=np.float32)
            reg_concat = np.array([], dtype=np.float32)
        outputs.append({
            "model_path": p,
            "cls_prob": cls_concat,
            "reg_pred": reg_concat
        })
    return outputs


def infer_worker(
    worker_id: int,
    device_id: int,
    model_paths: List[str],
    task_queue,
    result_queue,
    model_cfg: Dict,
    bs: int,
    num_worker: int,
    fusion_mode: str,
    seed: int
) -> None:
    try:
        fix_seed(seed + worker_id)
        device = resolve_device(device_id)
        cached_models = None
        if len(model_paths) == 1:
            cached_models = [
                build_model_from_checkpoint(
                    checkpoint_path=model_paths[0],
                    model_cfg=model_cfg,
                    fusion_mode=fusion_mode,
                    device=device
                )
            ]
            load_msg = "已常驻加载模型数=1"
        else:
            load_msg = f"将按 chunk 顺序加载模型数={len(model_paths)}"
        print(
            f"[INFO] worker={worker_id} 设备 {device} {load_msg} "
            f"(fusion_mode={fusion_mode})",
            flush=True
        )

        while True:
            task = task_queue.get()
            if task is None:
                break
            chunk_idx, reactions = task
            if cached_models is not None:
                outputs = infer_loaded_models_on_reactions(
                    models=cached_models,
                    model_paths=model_paths,
                    reactions=reactions,
                    device=device,
                    device_id=device_id,
                    bs=bs,
                    num_worker=num_worker,
                )
            else:
                outputs = []
                for p in model_paths:
                    model = build_model_from_checkpoint(
                        checkpoint_path=p,
                        model_cfg=model_cfg,
                        fusion_mode=fusion_mode,
                        device=device
                    )
                    outputs.extend(infer_loaded_models_on_reactions(
                        models=[model],
                        model_paths=[p],
                        reactions=reactions,
                        device=device,
                        device_id=device_id,
                        bs=bs,
                        num_worker=num_worker,
                    ))
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            result_queue.put({
                "status": "ok",
                "worker_id": worker_id,
                "chunk_idx": chunk_idx,
                "device": str(device),
                "outputs": outputs,
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception as e:
        result_queue.put({
            "status": "error",
            "worker_id": worker_id,
            "device_id": device_id,
            "error": repr(e),
        })


def build_device_tasks(
    paths: List[str],
    device_ids: List[int],
    models_per_device_parallel: int
) -> List:
    assigned = {d: [] for d in device_ids}
    for idx, p in enumerate(paths):
        assigned_device = device_ids[idx % len(device_ids)]
        assigned[assigned_device].append(p)

    device_tasks = []
    for d in device_ids:
        device_model_paths = assigned[d]
        if not device_model_paths:
            continue
        n_parallel = min(models_per_device_parallel, len(device_model_paths))
        if n_parallel == 1:
            device_tasks.append((d, device_model_paths))
            continue

        buckets = [[] for _ in range(n_parallel)]
        for idx, p in enumerate(device_model_paths):
            buckets[idx % n_parallel].append(p)
        for bucket in buckets:
            if bucket:
                device_tasks.append((d, bucket))
    return device_tasks


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
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200000,
        help="CSV 流式推理块大小；越小越省内存但调度开销越大"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--devices", type=str, default=None, help="逗号分隔设备ID，如 0,1,2；用于多卡并行推理")
    parser.add_argument(
        "--models_per_device_parallel",
        type=int,
        default=1,
        help="每张GPU上同时运行的模型推理进程数；默认1表示每张卡内模型顺序推理"
    )
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
    if args.models_per_device_parallel < 1:
        raise ValueError("--models_per_device_parallel 必须 >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk_size 必须 >= 1")
    device_ids = parse_device_ids(args.devices, args.device)
    print(f"[INFO] 推理设备: {device_ids}")
    print(f"[INFO] 每设备并发模型进程数: {args.models_per_device_parallel}")
    print(f"[INFO] CSV chunk_size: {args.chunk_size}")

    df_header = pd.read_csv(args.input, nrows=0)
    reaction_col = resolve_reaction_column(df_header, args.reaction_col)
    print(f"[INFO] 反应列: {reaction_col}")

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

    device_tasks = build_device_tasks(paths, device_ids, args.models_per_device_parallel)
    path_to_idx = {p: i for i, p in enumerate(paths)}
    max_parallel = len(device_tasks)
    per_proc_num_worker = args.num_worker // max_parallel if args.num_worker > 0 else 0
    print(
        f"[INFO] 流式并行推理模式: 并行进程={max_parallel}, "
        f"每进程DataLoader workers={per_proc_num_worker}"
    )
    for task_idx, (d, model_paths) in enumerate(device_tasks, start=1):
        print(
            f"[INFO] worker {task_idx - 1}: device={d}, 模型数={len(model_paths)}"
        )

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    task_queues = []
    workers = []

    try:
        for worker_id, (d, model_paths) in enumerate(device_tasks):
            task_queue = ctx.Queue(maxsize=1)
            proc = ctx.Process(
                target=infer_worker,
                args=(
                    worker_id,
                    d,
                    model_paths,
                    task_queue,
                    result_queue,
                    model_cfg,
                    args.bs,
                    per_proc_num_worker,
                    args.fusion_mode,
                    args.seed,
                ),
            )
            proc.start()
            task_queues.append(task_queue)
            workers.append(proc)

        wrote_header = False
        total_rows = 0
        chunk_count = 0
        for chunk_idx, df_chunk in enumerate(pd.read_csv(args.input, chunksize=args.chunk_size), start=1):
            chunk_count += 1
            if df_chunk[reaction_col].isna().any():
                raise ValueError(f"第 {chunk_idx} 个 chunk 的反应列 {reaction_col} 存在空值，请先清理")

            reactions = df_chunk[reaction_col].astype(str).tolist()
            if not reactions:
                continue

            print(
                f"[INFO] chunk {chunk_idx}: rows={len(reactions)}, "
                f"total_before={total_rows}",
                flush=True
            )
            for task_queue in task_queues:
                task_queue.put((chunk_idx, reactions))

            per_model_outputs = [None] * len(paths)
            received = 0
            while received < len(workers):
                try:
                    result = result_queue.get(timeout=60)
                except queue.Empty:
                    dead = [
                        f"worker={idx}, exitcode={proc.exitcode}"
                        for idx, proc in enumerate(workers)
                        if not proc.is_alive() and proc.exitcode is not None
                    ]
                    if dead:
                        raise RuntimeError("推理 worker 异常退出: " + "; ".join(dead))
                    continue

                if result.get("status") == "error":
                    raise RuntimeError(
                        f"worker={result.get('worker_id')} device={result.get('device_id')} "
                        f"推理失败: {result.get('error')}"
                    )
                if result["chunk_idx"] != chunk_idx:
                    raise RuntimeError(
                        f"收到错位 chunk 结果: expected={chunk_idx}, got={result['chunk_idx']}"
                    )

                received += 1
                for out in result["outputs"]:
                    per_model_outputs[path_to_idx[out["model_path"]]] = out

            if any(x is None for x in per_model_outputs):
                raise RuntimeError(f"第 {chunk_idx} 个 chunk 存在模型未完成推理，无法写出")

            for i, out in enumerate(per_model_outputs):
                df_chunk[f"model{i+1}_cls_prob"] = out["cls_prob"]
                df_chunk[f"model{i+1}_barrier"] = out["reg_pred"]

            df_chunk.to_csv(
                args.output,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            total_rows += len(df_chunk)
            print(f"[INFO] chunk {chunk_idx} 写出完成，累计行数={total_rows}", flush=True)

        if chunk_count == 0:
            raise ValueError("未读取到任何反应 SMILES")
    finally:
        for task_queue in task_queues:
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        for proc in workers:
            proc.join(timeout=30)
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)

    print(f"[INFO] 预测完成，保存至 {args.output}")
    print(f"[INFO] 使用 fusion_mode: {args.fusion_mode}")


if __name__ == "__main__":
    main()
