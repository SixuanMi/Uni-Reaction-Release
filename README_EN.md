# Uni-Reaction (Current Joint Pipeline)

This repository is currently trimmed to keep only the joint multitask pipeline used for:

- binary classification: `Is_elementary`
- regression: `Barrier`

The active model path is:

- `RAlignEncoder`
- `JointModel`
- `train_joint / eval_joint`

Other historical tasks and their model/training code have been removed from the main repo path.

## Current Model

The current joint model keeps:

- two fusion modes in the encoder:
  - `legacy`
  - `film`
- shared reactant/product encoder as the default behavior
- symmetric classification readout
- directional regression readout
- non-negative barrier output (`Softplus`)

At a high level:

1. Reactant and product graphs are encoded by `RAlignEncoder`.
2. Reactant and product are pooled separately.
3. Classification uses a symmetric reaction feature.
4. Regression uses an asymmetric reaction feature.

Current recommended setting for the main pipeline:

- `fusion_mode=film`
- shared reactant/product encoder enabled (default)

Key code files:

- `model/model.py`
- `model/block.py`
- `model/layers/RAlign.py`
- `model/layers/GATconv.py`
- `utils/Dataset.py`
- `utils/data_utils.py`
- `utils/training/training.py`

## Data Format

The current joint task expects CSV files with these columns:

- `Reaction`
- `Is_elementary`
- `Barrier`

Standard split layout:

- `train.csv`
- `val.csv`
- `test.csv`

`Reaction` should be mapped reaction SMILES in the form:

```text
reactants>>products
```

`Barrier` can be empty for samples without regression labels. Those samples still
participate in classification training, and regression loss is only computed on
rows with finite `Barrier`.

## Split Logic For K-Fold Training

Multi-fold training uses `prepare_joint_folds.py`.

Its split behavior is:

1. A single global test set is created first by `train_test_split(...)`.
2. This test split is fixed for all folds.
3. Only the remaining train/validation pool is further split by `StratifiedKFold`.
4. Therefore:
   - `test` is shared across all folds
   - `train` and `val` vary by fold

In other words, under the current implementation, **the test split is fixed, and only train/val change across folds**.

## Active Entry Points

The main scripts still in active use are:

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

## Training Usage

The shell wrappers in `scripts/` are the recommended way to launch training.

### Single Model

Train on one existing split directory that already contains:

- `train.csv`
- `val.csv`
- `test.csv`

Use:

```bash
./scripts/train_single.sh --data_path PATH_TO_SPLIT_DIR [EXTRA_ARGS...]
```

Example:

```bash
./scripts/train_single.sh \
  --data_path ./smoketest \
  --dim 192 \
  --n_layer 5 \
  --lr 2e-4 \
  --fusion_mode film
```

Notes:

- `base_log` defaults to `log_single_<timestamp>` if not provided.
- shared reactant/product encoder is enabled by default.
- pass `--no_share_reac_prod_encoder` only for ablation.

### Ensemble / Multi-Fold

This command first creates folds, then trains one model per fold.

Use:

```bash
PARALLEL_JOBS=5 GPU_IDS=0,1,2,3,4 ./scripts/train_ensemble.sh --data_path PATH_TO_DATA_DIR [EXTRA_ARGS...]
```

Example:

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

Important environment variables:

- `PARALLEL_JOBS`: number of folds trained at the same time
- `GPU_IDS`: comma-separated GPU ids, assigned to folds round-robin
- `N_FOLDS`: defaults to `5`
- `TEST_SIZE`: defaults to `0.1`
- `SEED`: defaults to `2025`
- `BASE_DIR`: defaults to `vote_run_<timestamp>`

Output layout:

- folds: `BASE_DIR/folds/fold_i`
- logs: `BASE_DIR/logs/fold_i/...`
- best checkpoint per run: `best_model.pth`

## Prediction Usage

### Single Model Evaluation

Recommended wrapper:

```bash
./scripts/predict_single.sh [ARGS...]
```

Two common ways to use it:

1. Directly provide explicit paths:

```bash
./scripts/predict_single.sh \
  --data_path PATH_TO_SPLIT_DIR \
  --checkpoint PATH_TO_BEST_MODEL \
  --output_path PATH_TO_OUTPUT_JSON \
  --dim 192 \
  --n_layer 5 \
  --fusion_mode film
```

2. Point it to an ensemble run directory:

```bash
./scripts/predict_single.sh \
  --main_dir vote_run_xxx \
  --fold 1 \
  --dim 192 \
  --n_layer 5 \
  --fusion_mode film
```

With `--main_dir`, defaults are:

- data: `main_dir/folds/fold_<fold>`
- checkpoint: latest `best_model.pth` found under `main_dir/logs/fold_<fold>/`
- output: `main_dir/logs/fold_<fold>/predict_result.json`

### Ensemble Evaluation

Recommended wrapper:

```bash
./scripts/predict_ensemble.sh [ARGS...]
```

Typical usage:

```bash
./scripts/predict_ensemble.sh \
  --main_dir vote_run_xxx \
  --dim 192 \
  --n_layer 5 \
  --fusion_mode film
```

Behavior with `--main_dir`:

- uses test data from `main_dir/folds/fold_1/test.csv`
- searches models under `main_dir/logs/**/best_model.pth`
- writes output to `main_dir/ensemble_result.json` unless `--output` is provided

You can also override this by passing:

- `--data_path`
- `--model_paths`
- `--output`

## Notes

- The repo is intentionally reduced to the current joint-task workflow, so older paper-wide functionality is no longer documented here.
