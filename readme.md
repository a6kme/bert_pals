### Abstract
We examine different approaches to improve the performance of the Bidirectional Encoder Representations from Transformers (BERT) on three downstream tasks: Sentiment Analysis, Paraphrase Detection, and Semantic Text Similarity (STS). Throughout our experimentation, a variety of different fine-tuning strategies and advanced techniques were leveraged including implementing Projected Attention Layers (PALs), multi-GPU training, Unsupervised Contrastive Learning of Sentence Embeddings (SimCSE), adding relational layers, hyperparameter tuning, and fine-tuning on additional datasets. We have found that a combination of PALs, unsupervised SimCSE, and additional relational layers resulted in the largest improvements in system accuracy.

### Report
[You can download the report here](cs224_report.pdf)

### Setup and Running
1. Setup a virtual environment `conda create -n cs224n_dfp python`
1. Activate the virutal environment `conda activate cs224n_dfp`
1. Install requirements `pip install -r requirements.txt`
1. Unzip `data.zip` which contains the data sources used for fine tuning and evaluation
1. Download the BERT Base model weights from BERT's official repository [Repo Link](https://github.com/google-research/bert/) || [File Link](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip)
1. Unzip the contents of the zip file in `uncased_L-12_H-768_A-12` folder
1. Convert the checkpoints to pytorch bin using below command
    ```
    transformers-cli convert --model_type bert \
    --tf_checkpoint uncased_L-12_H-768_A-12/bert_model.ckpt \
    --config uncased_L-12_H-768_A-12/bert_config.json \
    --pytorch_dump_output uncased_L-12_H-768_A-12/pytorch_model.bin
    ```
1. Fine tune the BERT model `src/multitask_classifier.py --fine-tune-mode full-model --lr 1e-5`

### Recommendations
1. It is recommended to run the training on a multi gpu cluster so that the training can run faster
