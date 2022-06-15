# Leveraging Intra-site Variability for Unsupervised  Site Adaptation (AIVA)

Pytorch implementation of our method AIVA (Submitted to MICCAI workshop).

## model scheme

![model scheme](images/schemeModel.png?raw=true "Title")

## Installation

* Install PyTorch from http://pytorch.org
* Install wandb using https://docs.wandb.ai/quickstart
* clone this repo
* cd repo folder
* ```pip3 install -r requirements.txt```

## Dataset

* Download CC359 from: https://www.ccdataset.com/download
* Download MultiSiteMri (msm) from: https://liuquande.github.io/SAML/
* point paths to the downloaded directories at paths.py
* run ```python3 -m dataset create_all_images_pickle```

## Pre-train Models

### cc359

```
python3 trainer.py --source {source} --target {source} --mode pretrain --gpu {device}
```

### msm

```
python3 trainer.py --source {source} --target {source} --mode pretrain --gpu {device} --msm
```

the results will be visible at https://wandb.ai/

## fine-tune model

### cc359

```
python3 trainer.py --source {source} --target {target} --mode clustering_finetune --gpu {device}
```

### msm

```
python3 trainer.py --source {source} --target {target} --mode clustering_finetune --gpu {device} --msm
```

the results will be visible at https://wandb.ai/