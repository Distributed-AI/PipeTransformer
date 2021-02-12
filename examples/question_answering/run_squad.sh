
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
PIPE_LEN=${11}

FREEZE_ALPHA=${12}
B_FREEZE=${13}
B_PIPE=${14}
B_DP=${15}
B_CACHE=${16}

python -m launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
main_qa.py \
--run_id $RUN_ID \
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
--pipe_len_at_the_beginning $PIPE_LEN \
--dataset squad_1.1 \
--data_dir "../../data/span_extraction/SQuAD_1.1/" \
--data_file "../../data/span_extraction/SQuAD_1.1/squad_1.1_data.pkl" \
--model_type bert \
--model_name bert-large-uncased \
--do_lower_case True \
--train_batch_size $BZ \
--eval_batch_size $BZ \
--max_seq_length 256 \
--learning_rate $LR \
--num_train_epochs 3 \
--output_dir "./output"
