# Transformer-based Fine-Grained Fungi Classification in an Open-Set Scenario

This repository is targeted towards solving the FungiCLEF 2022 (https://www.kaggle.com/competitions/fungiclef2022/) challenge. It is based on MMClassification (https://github.com/open-mmlab/mmclassification).

Pre-trained models can be found under Releases (https://github.com/wolfstefan/fungi-classification/releases).

## Usage

### Installation

```bash
conda create -n mmcls-fgvc python=3.8 -y
conda activate mmcls-fgvc
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-lts
pip install -e .
```

### Data

The challenge data has to be downloaded and put into _data/fungiclef2022/_.

### Training

```bash
bash tools/dist_train.sh configs/fungi/swin_large_b12x6-fp16_fungi+val_res_384_cb_epochs_6.py 6
```

### Inference

```bash
python tools/test_generate_result_pre-consensus_tta.py work_dirs/swin_large_b12x6-fp16_fungi+val_res_384_cb_epochs_6/swin_large_b12x6-fp16_fungi+val-test_res_384_cb_epochs_6.py work_dirs/swin_large_b12x6-fp16_fungi+val_res_384_cb_epochs_6/best_f1_score_epoch_6.pth results.csv
```