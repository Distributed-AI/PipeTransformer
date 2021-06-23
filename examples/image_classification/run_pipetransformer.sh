#!/usr/bin/env bash
export GLOO_SOCKET_IFNAME=eno1,eno2,ib0,lo

NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
CONFIG_FILE=$6
RUN_ID=$7

# sh run_pipetransformer.sh 8 1 0 10.0.198.185 11122 config/train_config_cifar100_no_freeze.yaml 1000
python -m launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
main_cv.py \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
--yaml_config_file $CONFIG_FILE \
--run_id $RUN_ID

