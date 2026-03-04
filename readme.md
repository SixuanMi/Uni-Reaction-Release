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

This README intentionally does not document detailed train/predict commands, because those entry scripts may be simplified further.

## Notes

- `co-diff/` is kept as a separate reference folder and is not part of the active main pipeline.
- The repo is intentionally reduced to the current joint-task workflow, so older paper-wide functionality is no longer documented here.
