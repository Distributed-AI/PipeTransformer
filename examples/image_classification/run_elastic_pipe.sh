#!/usr/bin/env bash
export GLOO_SOCKET_IFNAME=eno1,eno2,ib0,lo

# sh run_elastic_pipe.sh 8 2 1 192.168.11.2 11122 1 ib0 {args.lr} 400 imagenet /home/chaoyanghe/dataset/cv/imagenet 8 {args.freeze_strategy}
NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
IB=$6
IF_NAME=$7
LR=$8
BS=$9
DATASET=${10}
DATADIR=${11}
PIPE_LEN=${12}

FREEZE_ALPHA=${13}
B_FREEZE=${14}
B_PIPE=${15}
B_DP=${16}
B_CACHE=${17}

python -m launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
main_cv.py \
--$B_FREEZE \
--$B_PIPE \
--$B_DP \
--$B_CACHE \
--freeze_strategy_alpha $FREEZE_ALPHA \
--nnodes $NNODE \
--nproc_per_node=$NPROC_PER_NODE \
--node_rank $NODE_RANK \
--is_infiniband $IB \
--master_addr $MASTER_ADDR \
--if_name $IF_NAME \
--lr $LR \
--batch_size $BS \
--dataset $DATASET \
--data_dir $DATADIR \
--pipe_len_at_the_beginning $PIPE_LEN