# Uni-Reaction-Release

Official Implementation of paper:

[A Unified Chemical Reaction Representation Learning Framework for Reaction Condition Recommendation and Performance Prediction](https://arxiv.org/abs/2411.17629)

## Environment

install anaconda or miniconda, then set up the environment via

```shell
conda env create -f environment.yml
```

## Data, Checkpoints and Results

All the data used for model training and the checkpoints we have trained can be found through this [link](https://drive.google.com/drive/folders/1xruOB2ooBOaNGHQBT8_2ebGnwKMI2veG?usp=sharing). Below is a brief description of the format.

### Data

- **USPTO-Condition:** The folder contains two files. The `csv` file stores the training data, and the `json` file stores the reagent-index lookup table for the dataset.
- **USPTO-500MT:** The folder contains five `json` files. In addition to the training, validation, and test sets, we also provide a list of all reagents involved in the training labels and a list of chemical tokens.
- **Buchwald-Hartwig cross-coupling reaction:** (corresponding to the `bh` folder)  The folder contains ten random splits and four OOD splits. Each split folder has three files representing the training, validation, and test sets.
- **radical C–H functionalization:** (corresponding to the `hx` folder) The folder contains ten groups of random splits. The names of their subfolders indicate the random seeds used for the splits. Each split folder has three files representing the training, validation, and test sets.
- **chiral phosphoric acid-catalyzed thiol addition:** (corresponding to the `denmark` folder) The folder contains ten groups of random splits. The names of their subfolders indicate the random seeds used for the splits. Each split folder has three files representing the training, validation, and test sets.

### Checkpoints

- **USPTO-Condition:** The `model.pth` file represents the model weights, and the `pkl` file stores the reagent-index lookup table for easy metric calculation and validation.
- **USPTO-500MT:** The `model.pth` file represents the model weights, and the `pkl` file stores the reagent-index lookup table for easy metric calculation and validation.
- **Buchwald-Hartwig cross-coupling reaction:** (corresponding to the `cn` and `cn-nopretrain` folder) The folder contains checkpoints for each split. For OOD splits, we provide model weights trained with different random seeds.
- **radical C–H functionalization:** (corresponding to the `hx` folder) Each split folder contains the model weights corresponding to that split.
- **chiral phosphoric acid-catalyzed thiol addition:** (corresponding to the `dm` and `dm-nopretrain` folder) Each split folder contains the model weights corresponding to that split.

All checkpoints use the default parameters specified in the inference/training scripts. For the **Buchwald-Hartwig cross-coupling reaction** and **chiral phosphoric acid-catalyzed thiol addition** datasets, we provide two versions: one using a pretrained condition encoder and the other trained from scratch. You need to switch the model config as needed during inference. The model config is placed in the `config` folder.

### Results

We have uploaded the inference results for two sets of experiments that are relatively time-consuming, namely **USPTO-Condition** and **USPTO-500MT**, in the form of `json`.

## Data Preprocess

### USPTO-Condition

The Parrot project does not add atom-mapping for the reactions in USPTO-Condition. To address this, we need to use rxnmapper to add atom-mapping to USPTO-Condition. **To run the preprocess script, you need to create an environment that supports the operation of rxnmapper, following [its homepage](https://github.com/rxn4chemistry/rxnmapper).**

After obtaining the processed dataset from [Parrot](https://github.com/wangxr0526/Parrot), you can run the following commands to process the data. First, change the working directory:

```Shell
cd data_process_script/uspto_condition
```

Then, you need to execute the following script to add atom-mapping. Here, `$num_gpus` represents the number of GPUs required to add atom-mapping, `$batch_size` represents the batch size for rxnmapper inference, and `$share_size` represents the coarse-grained chunk size for the dataset. The script's execution logic is to divide the dataset into several chunks, write them into temporary files, and call another script to perform inference for different chunks on different devices. `$input_file` is the path to the `csv` file containing the processed data from the Parrot project, and `$output_file` is the path where the atom-mapped dataset will be output.

```Shell
python add_am_uspto_condition.py --num_gpus $num_gpus --batch_size $batch_size --share_size $share_size --input_file $input_file --output_file $output_file
```

Next, we also need to canonicalize the labels of the dataset and generate a reagent-index lookup table. `$input_file` is the path to the atom-mapped data, and `$vocab_path` is the path to the lookup table. If left blank, we will store the lookup table in `label_to_idx.json` in the same directory as the input file by default.

```Shell
python cano_output.py --file_path $input_file --vocab_path $vocab_path
```

Of course, we also provide preprocessed datasets (see **Data, Checkpoints and Results**).

### USPTO-500MT

Due to the presence of ionic compounds, a reagent may contain multiple components separated by '.', and to facilitate model training, these reagents should have a fixed order. Therefore, we have reorganized the sample labels for the dataset. After obtaining the raw data from T5Chem, you need to modify `DATA_DIR` in the two `py` files under the path `data_process_script/uspto_500mt` to the path of the raw data, and then run the following commands in sequence:

```Shell
cd data_process_scripts/uspto_500mt
python regenerate_reagent_uspto_500_mt.py
python generate_token_list_uspto_500_mt.py
```

Then you can obtain the processed dataset from the `data_process_scripts/uspto_500mt` folder. Of course, we also provide preprocessed datasets (see **Data, Checkpoints and Results**).

### Buchwald-Hartwig cross-coupling reaction

Since all the reactions in this dataset follow the same reaction template, we use a rule-based approach to generate atom-mapping for the reactions. We first obtain the raw data from [YieldBert](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Buchwald-Hartwig). Then, run the following command to get the data. Here, `$input_file` represents the path to the Excel file containing the raw data, and `$output_folder` is used to store the processed dataset. The script will output 10 random splits and four out-of-distribution (OOD) splits, which are aligned with the [YieldBert](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Buchwald-Hartwig) and other baselines.

```shell
python data_process_script/process_cn_yield.py --input_file $input_file --output_dir $output_folder
```

Of course, we also provide preprocessed datasets (see **Data, Checkpoints and Results**).

### radical C–H functionalization and chiral phosphoric acid-catalyzed thiol addition
The original data of these two datasets needs to be processed with rxnmapper to add atom-mapping and to perform random dataset splitting. Moreover, these two datasets share the same data splitting logic. Our data processing script is placed in the `data_process_script` folder, named `process_sel.ipynb`. After downloading the original data, you need to put the notebook and the downloaded data into the same folder, and then execute the corresponding cells according to the comments in the Jupyter Notebook to add atom-mapping and perform data splitting for the respective datasets. **The scripts needs to be executed under the environment of rxnmapper.** Of course, we also provide the preprocessed datasets (see [**Data, Checkpoints and Results**](##Data, Checkpoints and Results)).

## Training

### USPTO-Condition

To reproduce the training, use the following command for single-card training:

```shell
python train_uspto_condition.py --data_path $data_path --mapper_path $mapper_path
```

or the following command for single-machine multi-card training:

```shell
python train_uspto_condition_ddp.py --data_path $data_path --mapper_path $mapper_path
```

`$data_path` is the path to the processed `csv` file, and `$mapper_path` is the path to the reagent-index lookup table obtained during preprocessing.  The default parameters provided in the single-machine multi-card training code are the parameters used to train the open-source weights. You can also train your own version. Use the following commands to view all parameters and their meanings:

```shell
python train_uspto_condition.py -h
python train_uspto_condition_ddp.py -h
```

### USPTO-500MT

To reproduce the training, use the following command, where the `$data_path` here represents the folder containing the processed data.

```shell
python train_500mt_gen.py --data_path $data_path 
```

### Buchwald-Hartwig cross-coupling reaction

To reproduce the training, use the following command, where `$data_path` is the path to a specific data split of dataset and `$condition_config` is the path to the model config. We use `condition_config/cn/config_cn_no_pretrain_sep_gat.json` for the version without pretraining and `condition_config/cn/cn_config_pretrain_sep.json` for the version with a pretrained condition encoder.
```shell
python train_cn_full.py --data_path $data_path --base_log $base_log --condition_config $condition_config --condition_both
```

During the training, a logging directory named with the current timestamp will be generated in the folder `base_log`, where the checkpoint named `model.pth` and the training arguments named `log.json` are placed. To prevent confusion, you might need to set different `base_log` directories for different data splits. The default parameters provided in the code are those used to train the open-source weights, except for the non-pretrained version under OOD splits. **To reproduce experiments on the four OOD splits with a non-pretrained condition encoder, add the `--dim 64 --condition_config condition_config/cn/config_cn_no_pretrain_sep_gat_64_8_3.json`.**
```shell
python train_cn_full.py --data_path $data_path --base_log $base_log --condition_config condition_config/cn/config_cn_no_pretrain_sep_gat_64_8_3.json --condition_both --dim 64
```

### radical C–H functionalization

To reproduce the training, use the following command, where `$data_path` is the path to a specific data split of dataset.

```shell
python train_hx.py --data_path $data_path --base_log $base_log 
```

During the training, a logging directory named with the current timestamp will be generated in the folder `base_log`. To prevent confusion, you might need to set different `base_log` directories for different data splits.

### chiral phosphoric acid-catalyzed thiol addition

To reproduce the training, use the following command, where `$data_path` is the path to a specific data split of dataset and `$condition_config` is the path to the model config. We use `condition_config\dm\config_dm_no_pretrain_gat.json` for the version without pretraining and `condition_config\dm\config_dm_no_pretrain_gat.json` for the version with a pretrained condition encoder.
```shell
python train_dm.py --data_path $data_path --condition_both --condition_config condition_config/dm/config_dm_no_pretrain_gat.json
```

During the training, a logging directory named with the current timestamp will be generated in the folder `base_log`. To prevent confusion, you might need to set different `base_log` directories for different data splits.

## Inference and Evaluation

### USPTO-Condition

To inference the prediction using a certain checkpoint, use the following command, where `$data_path` is the path of the processed dataset, `$checkpoint` is the path of checkpoint, `$token_ckpt` is the path of the corresponding `pkl` file for the reagent-index lookup table and `$output_path` is the output file path. 

```shell
python inference_condition.py --data_path $data_path --token_ckpt $token_ckpt --checkpoint $checkpoint --output_file $output_path
```

The provided default values of args are those of the provided checkpoints. To adapt to different structures and hardware, you may need to modify the other parameters. Use the following commands to view the relevant parameters and their meanings.

```shell
python inference_condition.py -h
```

To evaluate the results, use the following command, where `$input_file` is path of the `json` file obtained via the inference scripts and `$beam` is the beam size for beam search during the inference.

```shell
python evaluate_condition.py --file $input_file --beam $beam
python evaluate_pred_split.py --file $input_file --beam $beam
```

### USPTO-500MT

To inference the prediction using a certain checkpoint, use the following command, where `$data_path` is the path of the processed dataset, `$checkpoint` is the path of checkpoint, `$token_ckpt` is the path of the corresponding `pkl` file for the smiles tokenizer and `$output_path` is the output file path. 

```shell
python inference_uspto_500mt.py --data_path $data_path --token_ckpt $token_ckpt --checkpoint $checkpoint --output_file $output_path
```

To evaluate the results, use the following command, where `$input_file` is path of the `json` file obtained via the inference scripts and `$beam` is the beam size for beam search during the inference.

```shell
python evaluate_500mt.py --file $input_file --beam $beam
```

### Buchwald-Hartwig cross-coupling reaction and chiral phosphoric acid-catalyzed thiol addition

To inference and evaluate the result, use the following command, where `$data_path` is the path of the specific data split of dataset need to evaluate, `$condition_config` is the path to the model config, `$checkpoint` is the path to the checkpoint and `$output_path` is the path to store the prediction and ground truth. 

```shell
python predict_cn.py/predict_dm.py --data_path $data_path --condition_config $condition_config --checkpoint $checkpoint --output $output_path --condition_both
```

`predict_cn.py` is for Buchwald-Hartwig cross-coupling reaction dataset and `predict_dm` is for chiral phosphoric acid-catalyzed thiol addition. `condition_config/cn/cn_config_pretrain_sep.json` and `condition_config/cn/config_cn_no_pretrain_sep_gat.json` is for the version with pretrained/non-pretrained condition encoder for Buchwald-Hartwig cross-coupling reaction dataset, respectively. `condition_config\dm\config_dm_no_pretrain_gat.json` and `condition_config\dm\config_dm_no_pretrain_gat.json` is for the version with pretrained/non-pretrained condition encoder for chiral phosphoric acid-catalyzed thiol addition dataset, respectively.  **To perform the inference on the provided checkpoint of non-pretrained version on Buchwald-Hartwig cross-coupling reaction dataset's OOD split, add `--dim 64 --condition_config condition_config/cn/config_cn_no_pretrain_sep_gat_64_8_3.json`**

### radical C–H functionalization

To inference and evaluate the result, use the following command, where `$data_path` is the path of the specific data split of dataset need to evaluate, `$condition_config` is the path to the model config and `$output_path` is the path to store the prediction and ground truth. 

```shell
python predict_hx.py --data_path $data_path --checkpoint $checkpoint --output $output_path
```

