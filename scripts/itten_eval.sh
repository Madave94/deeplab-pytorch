#!/bin/bash

# import language settings to run on itten-server
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# 0. Dataset
DATASET=S2DS

# 1. Evaluate the model on val set

python main.py test \
-c configs/${DATASET}.yaml \
-m data/models/${DATASET}/deeplabv2_resnet101_msc/*/checkpoint_final.pth

# 3. Re-evaluate the model with CRF post-processing
python main.py crf \
-c configs/${DATASET}.yaml
