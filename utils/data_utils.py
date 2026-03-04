import os
import random

import numpy as np
import pandas as pd
import torch

from .Dataset import JointDataset


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


def load_joint_data(data_path):
    train_set = load_joint_data_one(data_path, 'train')
    val_set = load_joint_data_one(data_path, 'val')
    test_set = load_joint_data_one(data_path, 'test')
    return train_set, val_set, test_set


def load_joint_data_one(data_path, part):
    csv_path = os.path.join(data_path, f'{part}.csv')
    data = pd.read_csv(csv_path)

    reactions = []
    is_elementary = []
    barrier = []
    for _, row in data.iterrows():
        reactions.append(row['Reaction'])
        is_elementary.append(int(row['Is_elementary']))
        barrier_val = row['Barrier']
        barrier.append(float(barrier_val) if pd.notna(barrier_val) else float('nan'))

    return JointDataset(
        reactions=reactions,
        is_elementary=is_elementary,
        barrier=barrier
    )
