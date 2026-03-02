import pandas as pd
import random
import numpy as np
import torch
import os
import json
from tqdm import tqdm

from .Dataset import (
    CNYieldDataset, SelDataset, ReactionPredDataset, JointDataset, 
    ReactionSeqInferenceDataset
)

from .tokenlizer import Tokenizer, smi_tokenizer


def load_sel(data_path, condition_type='pretrain', has_reag=True):
    train_set = load_sel_one(data_path, 'train', condition_type, has_reag)
    val_set = load_sel_one(data_path, 'val', condition_type, has_reag)
    test_set = load_sel_one(data_path, 'test', condition_type, has_reag)
    return train_set, val_set, test_set


def load_sel_one(data_path, part, condition_type='pretrain', has_reag=True):
    train_x = pd.read_csv(os.path.join(data_path, f'{part}.csv'))
    rxn, out, catalyst = [[] for _ in range(3)]
    for i, x in train_x.iterrows():
        rxn.append(x['mapped_rxn'])
        out.append(x['Output'])
        if has_reag:
            catalyst.append(x['Catalyst'])

    return SelDataset(
        reactions=rxn, catalyst=catalyst if len(catalyst) else None,
        labels=out, condition_type=condition_type
    )


def load_cn_yield(data_path, condition_type='pretrain'):
    train_set = load_cn_yield_one(data_path, 'train', condition_type)
    val_set = load_cn_yield_one(data_path, 'val', condition_type)
    test_set = load_cn_yield_one(data_path, 'test', condition_type)
    return train_set, val_set, test_set


def load_cn_yield_one(data_path, part, condition_type='pretrain'):
    train_x = pd.read_csv(os.path.join(data_path, f'{part}.csv'))
    rxn, out, ligand, base, additive, catalyst = [[] for _ in range(6)]
    for i, x in train_x.iterrows():
        rxn.append(x['mapped_rxn'])
        out.append(x['Output'])
        ligand.append(x['Ligand'])
        base.append(x['Base'])
        additive.append(x['Additive'])
        catalyst.append(x['catalyst'])

    return CNYieldDataset(
        reactions=rxn, ligand=ligand, catalyst=catalyst, base=base,
        additive=additive, labels=out,  condition_type=condition_type
    )


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(module):
    total_params = 0
    trainable_params = 0
    for param in module.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return total_params, trainable_params


def load_uspto_mt_500_gen(data_path, remap=None, part=None):
    if remap is None:
        with open(os.path.join(data_path, 'all_tokens.json')) as F:
            reag_list = json.load(F)
        remap = Tokenizer(reag_list, {'<UNK>', '<CLS>', '<END>', '<PAD>', '`'})

    with open(os.path.join(data_path, 'all_reagents.json')) as F:
        INFO = json.load(F)
    reag_order = {k: idx for idx, k in enumerate(INFO)}

    rxns, px = [[], [], []], 0
    labels = [[], [], []]
    if part is None:
        iterx = ['train.json', 'val.json', 'test.json']
    else:
        iterx = [f'{part}.json']
    for infos in iterx:
        F = open(os.path.join(data_path, infos))
        setx = json.load(F)
        F.close()
        for lin in setx:
            rxns[px].append(lin['new_mapped_rxn'])
            lin['reagent_list'].sort(key=lambda x: reag_order[x])
            lbs = []
            for tdx, x in enumerate(lin['reagent_list']):
                if tdx > 0:
                    lbs.append('`')
                lbs.extend(smi_tokenizer(x))
            labels[px].append(lbs)
        px += 1

    if part is not None:
        return ReactionPredDataset(
            reactions=rxns[0], labels=labels[0],
            cls_id='<CLS>', end_id='<END>'
        ), remap

    train_set = ReactionPredDataset(
        reactions=rxns[0], labels=labels[0],
        cls_id='<CLS>', end_id='<END>'
    )

    val_set = ReactionPredDataset(
        reactions=rxns[1], labels=labels[1],
        cls_id='<CLS>', end_id='<END>'
    )

    test_set = ReactionPredDataset(
        reactions=rxns[2], labels=labels[2],
        cls_id='<CLS>', end_id="<END>"
    )

    return train_set, val_set, test_set, remap


def load_uspto_mt500_inference(data_path, remap):
    rxns, labels = [], []
    with open(data_path) as F:
        setx = json.load(F)
        for lin in setx:
            rxns.append(lin['new_mapped_rxn'])
            labels.append('.'.join(lin['reagent_list']))

    dataset = ReactionSeqInferenceDataset(rxns, labels, True)
    return dataset


def check_early_stop(*args):
    answer = True
    for x in args:
        answer &= all(t <= x[0] for t in x[1:])
    return answer


def load_uspto_condition(data_path, mapper_path='', verbose=True, mapper=None):
    raw_info = pd.read_csv(data_path)
    raw_info = raw_info.fillna('')
    raw_info = raw_info.to_dict('records')

    if mapper is None:
        with open(mapper_path) as Fin:
            mapper = json.load(Fin)
    mapper['<CLS>'] = mapper.get('<CLS>', len(mapper))

    all_datas = {
        'train_reac': [], 'train_label': [],
        'val_reac': [], 'val_label': [],
        'test_reac': [], 'test_label': []
    }

    iterx = tqdm(raw_info) if verbose else raw_info
    for i, element in enumerate(iterx):
        rxn_type = element['dataset']
        all_datas[f'{rxn_type}_reac'].append(element['mapped_rxn'])
        labels = [
            mapper[element['catalyst1']],
            mapper[element['solvent1']], mapper[element['solvent2']],
            mapper[element['reagent1']], mapper[element['reagent2']]
        ]
        all_datas[f'{rxn_type}_label'].append(labels)

    train_set = ReactionPredDataset(
        reactions=all_datas['train_reac'],
        labels=all_datas['train_label'], cls_id=mapper['<CLS>'],

    )
    val_set = ReactionPredDataset(
        reactions=all_datas['val_reac'],
        labels=all_datas['val_label'], cls_id=mapper['<CLS>']
    )

    test_set = ReactionPredDataset(
        reactions=all_datas['test_reac'],
        labels=all_datas['test_label'], cls_id=mapper["<CLS>"]
    )

    return train_set, val_set, test_set, mapper


def load_uspto_condition_inference(data_path, mapper):
    raw_info = pd.read_csv(data_path)
    raw_info = raw_info.fillna('')
    raw_info = raw_info.to_dict('records')
    reac, all_labels = [], []

    for i, element in enumerate(tqdm(raw_info)):
        if element['dataset'] != 'test':
            continue
        reac.append(element['mapped_rxn'])
        labels = [
            mapper[element['catalyst1']],
            mapper[element['solvent1']], mapper[element['solvent2']],
            mapper[element['reagent1']], mapper[element['reagent2']]
        ]
        all_labels.append(labels)

    dataset = ReactionSeqInferenceDataset(reac, all_labels, True)
    return dataset


def load_joint_data(data_path, use_local_pe=False):
    """加载训练/验证/测试集的联合数据（分类+回归）"""
    train_set = load_joint_data_one(data_path, 'train', use_local_pe=use_local_pe)
    val_set = load_joint_data_one(data_path, 'val', use_local_pe=use_local_pe)
    test_set = load_joint_data_one(data_path, 'test', use_local_pe=use_local_pe)
    return train_set, val_set, test_set

def load_joint_data_one(data_path, part, use_local_pe=False):
    """加载单个数据集（train/val/test），适配JointDataset"""
    # 读取CSV文件（如train.csv、val.csv、test.csv）
    csv_path = os.path.join(data_path, f'{part}.csv')
    data = pd.read_csv(csv_path)
    
    # 从CSV中提取所需列（与JointDataset参数对应）
    reactions = []  # 反应数据（对应Reaction列）
    is_elementary = []  # 分类标签（对应Is_elementary列）
    barrier = []  # 回归标签（对应Barrier列，可能含NaN）
    
    for _, row in data.iterrows():
        # 提取反应数据（Reaction列）
        reactions.append(row['Reaction'])
        # 提取分类标签（Is_elementary列，转换为整数）
        is_elementary.append(int(row['Is_elementary']))
        # 提取回归标签（Barrier列，保留浮点数，允许NaN）
        barrier_val = row['Barrier']
        # 处理可能的空值（转换为NaN，确保后续损失函数能识别）
        barrier.append(float(barrier_val) if pd.notna(barrier_val) else float('nan'))
    
    # 初始化JointDataset（参数与类定义一致：reactions, is_elementary, barrier）
    return JointDataset(
        reactions=reactions,
        is_elementary=is_elementary,
        barrier=barrier,
        use_local_pe=use_local_pe
    )
    
# def load_joint_data_one(data_path, part):
#     """加载单个数据集（train/val/test），并对回归标签（Barrier）进行标准化处理"""
#     # 读取CSV文件
#     csv_path = os.path.join(data_path, f'{part}.csv')
#     data = pd.read_csv(csv_path)
    
#     # 提取基础数据
#     reactions = []  # 反应数据（Reaction列）
#     is_elementary = []  # 分类标签（Is_elementary列）
#     raw_barrier = []  # 原始回归标签（用于标准化）
    
#     for _, row in data.iterrows():
#         # 提取反应SMILES
#         reactions.append(row['Reaction'])
#         # 提取分类标签（转为整数）
#         is_elementary.append(int(row['Is_elementary']))
#         # 提取原始Barrier值（保留NaN）
#         barrier_val = row['Barrier']
#         raw_barrier.append(float(barrier_val) if pd.notna(barrier_val) else float('nan'))
    
#     # --------------------------
#     # 回归标签标准化（Z-score）
#     # --------------------------
#     # 定义标准化参数保存路径
#     mean_path = os.path.join(data_path, 'barrier_mean.npy')
#     std_path = os.path.join(data_path, 'barrier_std.npy')
    
#     if part == 'train':
#         # 训练集：计算均值和标准差（仅用非NaN值）
#         valid_barriers = [x for x in raw_barrier if not np.isnan(x)]
#         if len(valid_barriers) == 0:
#             raise ValueError("训练集中没有有效的Barrier值（全为NaN），无法进行标准化")
        
#         barrier_mean = np.mean(valid_barriers)
#         barrier_std = np.std(valid_barriers)
        
#         # 保存标准化参数（供验证集/测试集使用）
#         np.save(mean_path, barrier_mean)
#         np.save(std_path, barrier_std)
#         print(f"[标准化参数] 训练集Barrier均值: {barrier_mean:.4f}, 标准差: {barrier_std:.4f}")
    
#     else:
#         # 验证集/测试集：使用训练集的标准化参数
#         if not (os.path.exists(mean_path) and os.path.exists(std_path)):
#             raise FileNotFoundError(f"未找到标准化参数文件，请先运行训练集加载（需在{data_path}生成barrier_mean.npy和barrier_std.npy）")
        
#         barrier_mean = np.load(mean_path)
#         barrier_std = np.load(std_path)
    
#     # 对原始Barrier进行标准化（NaN值保持不变）
#     barrier = []
#     for val in raw_barrier:
#         if np.isnan(val):
#             barrier.append(float('nan'))  # 保留NaN（无标签）
#         else:
#             # Z-score标准化：(x - mean) / std
#             normalized = (val - barrier_mean) / barrier_std
#             barrier.append(normalized)
    
#     # 返回标准化后的数据集
#     return JointDataset(
#         reactions=reactions,
#         is_elementary=is_elementary,
#         barrier=barrier
#     )
