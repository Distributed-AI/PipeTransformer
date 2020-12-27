#!/usr/bin/env bash

LR=$1
BS=$2
DATASET=$3
DATADIR=$4
IS_GPU=$5


python main_single_gpu.py \
--lr $LR \
--batch_size $BS \
--dataset $DATASET \
--data_dir $DATADIR \
--is_gpu $IS_GPU