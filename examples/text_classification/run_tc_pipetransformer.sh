NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
IB=$6
IF_NAME=$7
LR=$8
BZ=$9
RUN_ID=${10}
B_FREEZE=${11}
PIPE_LEN=${12}

python -m launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
main_tc.py \
--run_id $RUN_ID \
--$B_FREEZE \
--nnodes $NNODE \
--nproc_per_node=$NPROC_PER_NODE \
--node_rank $NODE_RANK \
--is_infiniband $IB \
--master_addr $MASTER_ADDR \
--if_name $IF_NAME \
--pipe_len_at_the_beginning $PIPE_LEN \
--dataset "sst_2" \
--data_dir "../../data/text_classification/SST-2/trees" \
--data_file "../../data/text_classification/SST-2/sst_2_data.pkl" \
--model_type bert \
--model_name bert-base-uncased \
--do_lower_case True \
--train_batch_size $BZ \
--eval_batch_size $BZ \
--max_seq_length 256 \
--learning_rate $LR \
--num_train_epochs 3 \
--output_dir "./output"
