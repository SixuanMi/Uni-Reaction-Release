# Uni-Reaction（当前联合任务主链路）

英文说明见 [`README_EN.md`](./README_EN.md)。

当前仓库已经裁剪为只保留联合多任务主链路，任务包括：

- 二分类：`Is_elementary`
- 回归：`Barrier`

当前仍在使用的主模型路径是：

- `RAlignEncoder`
- `JointModel`
- `train_joint / eval_joint`

历史任务及其对应的模型、训练逻辑已从主仓库路径中清理。

## 当前模型

当前联合模型保留了以下关键设计：

- 编码器中保留两种融合方式：
  - `legacy`
  - `film`
- 反应物/产物编码器默认共享参数
- 分类分支采用对称读出
- 回归分支采用保留方向性的读出
- 回归输出使用非负约束（`Softplus`）

整体流程如下：

1. 反应物图和产物图先经过 `RAlignEncoder` 编码。
2. 反应物和产物分别做池化。
3. 分类分支使用对称的反应级特征。
4. 回归分支使用非对称的反应级特征。

当前主链路推荐配置：

- `fusion_mode=film`
- 开启反应物/产物编码器共享（默认即开启）

核心代码文件：

- `model/model.py`
- `model/block.py`
- `model/layers/RAlign.py`
- `model/layers/GATconv.py`
- `utils/Dataset.py`
- `utils/data_utils.py`
- `utils/training/training.py`

## 数据格式

当前联合任务要求 CSV 至少包含以下列：

- `Reaction`
- `Is_elementary`
- `Barrier`

标准数据切分目录结构为：

- `train.csv`
- `val.csv`
- `test.csv`

其中 `Reaction` 需要是带 atom-map 的反应 SMILES，格式为：

```text
reactants>>products
```

`Barrier` 可以为空。对于没有回归标签的样本：

- 仍然参与分类训练
- 回归损失只在 `Barrier` 为有限数值的样本上计算

## 多折划分逻辑

多折训练使用 `prepare_joint_folds.py`。

当前划分方式是：

1. 先通过一次 `train_test_split(...)` 切出全局测试集。
2. 这份测试集在所有 fold 中保持固定。
3. 只对剩余的训练/验证池再做 `StratifiedKFold`。
4. 因此：
   - `test` 在所有 fold 中相同
   - `train` 和 `val` 会随 fold 变化

也就是说，当前实现中，**测试集是固定的，只有 train/val 会在不同 fold 中变化**。

## 当前入口脚本

当前仍在使用的主要入口包括：

- `train_elementary.py`
- `prepare_joint_folds.py`
- `scripts/train_single.sh`
- `scripts/train_ensemble.sh`
- `predict_elementary.py`
- `predict_ensemble.py`
- `scripts/predict_single.sh`
- `scripts/predict_ensemble.sh`
- `vote_infer_unlabeled.py`
- `analyze_swap_sensitivity.py`
- `active_select.py`

## 训练使用方法

推荐优先使用 `scripts/` 下的 shell 包装脚本。

### 单模型训练

用于训练一个已经切好分的数据目录，该目录应包含：

- `train.csv`
- `val.csv`
- `test.csv`

用法：

```bash
./scripts/train_single.sh --data_path PATH_TO_SPLIT_DIR [EXTRA_ARGS...]
```

示例：

```bash
./scripts/train_single.sh \
  --data_path ./smoketest \
  --dim 192 \
  --n_layer 5 \
  --lr 2e-4 \
  --fusion_mode film
```

说明：

- 如果不显式传 `base_log`，默认日志目录为 `log_single_<timestamp>`
- 反应物/产物编码器共享默认开启
- 只有做消融实验时才需要传 `--no_share_reac_prod_encoder`

### 多模型 / 多折训练

该命令会先生成 folds，再为每个 fold 启动一个训练。

用法：

```bash
PARALLEL_JOBS=5 GPU_IDS=0,1,2,3,4 ./scripts/train_ensemble.sh --data_path PATH_TO_DATA_DIR [EXTRA_ARGS...]
```

示例：

```bash
PARALLEL_JOBS=5 GPU_IDS=0,1,2,3,4 ./scripts/train_ensemble.sh \
  --data_path ../dataset/ready_v6_t1xTruexTB_FalsexTB_stereo_cycle0/ \
  --dim 192 \
  --n_layer 5 \
  --lr 2e-4 \
  --num_worker 32 \
  --epoch 200 \
  --fusion_mode film
```

常用环境变量：

- `PARALLEL_JOBS`：同时训练的 fold 数量
- `GPU_IDS`：逗号分隔的 GPU 编号，按 fold 轮询分配
- `N_FOLDS`：默认 `5`
- `TEST_SIZE`：默认 `0.1`
- `SEED`：默认 `2025`
- `BASE_DIR`：默认 `vote_run_<timestamp>`

输出目录结构：

- folds：`BASE_DIR/folds/fold_i`
- logs：`BASE_DIR/logs/fold_i/...`
- 每次训练保存的最佳权重：`best_model.pth`

## 预测使用方法

### 单模型评估

推荐入口：

```bash
./scripts/predict_single.sh [ARGS...]
```

常见有两种用法。

1. 显式传入完整路径：

```bash
./scripts/predict_single.sh \
  --data_path PATH_TO_SPLIT_DIR \
  --checkpoint PATH_TO_BEST_MODEL \
  --output_path PATH_TO_OUTPUT_JSON \
  --dim 192 \
  --n_layer 5 \
  --fusion_mode film
```

2. 直接指向一个多折训练输出目录：

```bash
./scripts/predict_single.sh \
  --main_dir vote_run_xxx \
  --fold 1 \
  --dim 192 \
  --n_layer 5 \
  --fusion_mode film
```

当使用 `--main_dir` 时，默认规则为：

- 数据路径：`main_dir/folds/fold_<fold>`
- 权重路径：`main_dir/logs/fold_<fold>/` 下找到的最新 `best_model.pth`
- 输出路径：`main_dir/logs/fold_<fold>/predict_result.json`

### 多模型集成评估

推荐入口：

```bash
./scripts/predict_ensemble.sh [ARGS...]
```

典型用法：

```bash
./scripts/predict_ensemble.sh \
  --main_dir vote_run_xxx \
  --dim 192 \
  --n_layer 5 \
  --fusion_mode film
```

当使用 `--main_dir` 时：

- 测试数据默认来自 `main_dir/folds/fold_1/test.csv`
- 模型默认从 `main_dir/logs/**/best_model.pth` 搜索
- 输出默认写到 `main_dir/ensemble_result.json`，除非显式传 `--output`

也可以手动覆盖：

- `--data_path`
- `--model_paths`
- `--output`

## 备注

- 当前仓库刻意收敛为联合任务工作流，因此旧版论文中的其他功能不再在这里维护。
