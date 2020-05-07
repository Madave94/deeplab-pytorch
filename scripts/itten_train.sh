#!/bin/bash

# 0. S2DS
DATASET=S2DS

# import language settings to run on itten-server
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# 1. Train DeepLab v2 on ${DATASET}
python main.py train \
-c configs/${DATASET}.yaml

# Trained models are saved into
#   data/models/${DATASET}/deeplabv2_resnet101_msc/*/checkpoint_5000.pth
#   data/models/${DATASET}/deeplabv2_resnet101_msc/*/checkpoint_10000.pth
#   data/models/${DATASET}/deeplabv2_resnet101_msc/*/checkpoint_15000.pth
#   ...

# Tensorboard logs are in data/logs.