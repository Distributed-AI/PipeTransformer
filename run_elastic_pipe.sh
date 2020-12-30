#!/usr/bin/env bash
export GLOO_SOCKET_IFNAME=eno1,eno2,ib0,lo

NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
IB=$6
LR=$7
BS=$8
DATASET=$9
DATADIR=${10}
PIPE_LEN=${11}

python -m torch.distributed.launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
main.py \
--is_infiniband $IB \
--lr $LR \
--batch_size $BS \
--dataset $DATASET \
--data_dir $DATADIR \
--pipe_len_at_the_beginning $PIPE_LEN