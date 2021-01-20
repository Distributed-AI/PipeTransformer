NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
IB=$6
IF_NAME=$7

python -m launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
main_tc.py \
--nnodes $NNODE \
--nproc_per_node=$NPROC_PER_NODE \
--node_rank $NODE_RANK \
--is_infiniband $IB \
--master_addr $MASTER_ADDR \
--if_name $IF_NAME \
--dataset "sst_2" \
--data_dir "../../data/text_classification/SST-2/trees" \
--data_file "../../data/text_classification/SST-2/sst_2_data.pkl" \
--model_type bert \
--model_name bert-base-uncased \
--do_lower_case True \
--train_batch_size 32 \
--eval_batch_size 32 \
--max_seq_length 512 \
--learning_rate 3e-5 \
--num_train_epochs 5 \
--output_dir "./output"
