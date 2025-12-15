# Uni-Reaction

Add multitask learning of Official Implementation of paper:

[A Unified Chemical Reaction Representation Learning Framework for Reaction Condition Recommendation and Performance Prediction](https://arxiv.org/abs/2411.17629)

## Environment

install anaconda or miniconda, then set up the environment via

```shell
conda env create -f environment.yml
```

## 快速使用（投票 + 主动学习流程）

- 多模型训练（K 折投票）  
  `bash run_vote_train.sh <原始数据目录> [其他 train_elementary 参数]`  
  会在 `vote_run_<timestamp>/` 下生成折分数据（folds）和各折日志/模型（logs）。

- 投票评估（已有折分模型）  
  `python run_vote_predict.py --main_dir vote_run_<timestamp> --output ensemble_result.json --dim ... --heads ...`  
  如需自定义测试集或模型列表，可用 `--data_path` / `--model_paths` 覆盖。

- 对无标签反应做多模型推理（获取每模型分类概率+能垒）  
  `python vote_infer_unlabeled.py --input reactions.csv --main_dir vote_run_<timestamp> --output vote_preds.csv --dim ...`

- 主动学习样本选择（分类不确定性优先策略）  
  `python active_select.py --input vote_preds.csv --top_n 100 --output al_selected.csv`

- 单模型训练快捷方式  
  `bash run_train.sh [其他 train_elementary 参数]`（默认数据路径 `../dataset/ready_v3_t1xTrueDFT_FalseDFT`）