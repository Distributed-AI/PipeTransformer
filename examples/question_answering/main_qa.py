import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
# this is a temporal import, we will refactor FedML as a package installation
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from pipe_transformer.config_args import ConfigArgs
from pipe_transformer.pipe_transformer import PipeTransformer

from pipe_transformer.data.qa_data_manager import QADatasetManager
from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering

from examples.question_answering.model_args import QuestionAnsweringArgs

from examples.question_answering.question_answering_trainer import QuestionAnsweringTrainer


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # PipeTransformer related
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--nnodes", type=int, default=2)

    parser.add_argument("--nproc_per_node", type=int, default=8)

    parser.add_argument("--node_rank", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--global_rank", type=int, default=0)

    parser.add_argument("--master_addr", type=str, default="192.168.11.2")

    parser.add_argument("--master_port", type=int, default=22222)

    parser.add_argument("--if_name", type=str, default="lo")

    parser.add_argument("--pipe_len_at_the_beginning", default=4, type=int,
                        help="pipe_len_at_the_beginning")

    parser.add_argument("--is_infiniband", default=1, type=int,
                        help="is_infiniband")

    parser.add_argument("--num_chunks_of_micro_batches", default=1 * 8, type=int,
                        help="num_chunks_of_micro_batches")

    parser.add_argument("--freeze_strategy_alpha", type=float, default=0.33)

    parser.add_argument('--freeze', dest='b_freeze', action='store_true')
    parser.add_argument('--no_freeze', dest='b_freeze', action='store_false')
    parser.set_defaults(b_freeze=True)

    parser.add_argument('--auto_pipe', dest='b_auto_pipe', action='store_true')
    parser.add_argument('--do_auto_pipe', dest='b_auto_pipe', action='store_false')
    parser.set_defaults(b_auto_pipe=True)

    parser.add_argument('--auto_dp', dest='b_auto_dp', action='store_true')
    parser.add_argument('--no_auto_dp', dest='b_auto_dp', action='store_false')
    parser.set_defaults(b_auto_dp=True)

    parser.add_argument('--cache', dest='b_cache', action='store_true')
    parser.add_argument('--no_cache', dest='b_cache', action='store_false')
    parser.set_defaults(b_cache=True)

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # Data related
    parser.add_argument('--dataset', type=str, default='squad_1.1', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='../../data/text_classification/20Newsgroups/20news-18828',
                        help='data directory')

    parser.add_argument('--data_file', type=str, default='data/data_loaders/squad_1.1_data_loader.pkl',
                        help='data pickle file')

    parser.add_argument('--eval_data_file', type=str, default='../../data/span_extraction/SQuAD_1.1/dev-v1.1.json',
                        help='this argument is set up for using official script to evaluate the model')

    # Model related
    parser.add_argument('--model_type', type=str, default='distilbert', metavar='N',
                        help='transformer model type')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', metavar='N',
                        help='transformer model name')
    parser.add_argument('--do_lower_case', type=bool, default=False, metavar='N',
                        help='transformer model name')

    # Learning related

    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=128, metavar='N',
                        help='maximum sequence length (default: 128)')

    parser.add_argument('--learning_rate', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                        help='L2 penalty')

    parser.add_argument('--num_train_epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')
    parser.add_argument('--n_gpu', type=int, default=1, metavar='EP',
                        help='how many gpus will be used ')
    parser.add_argument('--fp16', default=False, action="store_true",
                        help='if enable fp16 for training')
    parser.add_argument('--manual_seed', type=int, default=42, metavar='N',
                        help='random seed')

    # IO realted

    parser.add_argument('--output_dir', type=str, default="/tmp/squad_1.1", metavar='N',
                        help='path to save the trained results and ckpts')

    args = parser.parse_args()

    return args


def create_model(args):
    # create model, tokenizer, and model config (HuggingFace style)
    MODEL_CLASSES = {
        "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name, **args.config)
    model = model_class.from_pretrained(model_name, config=config)
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=args.do_lower_case)
    return config, model, tokenizer


def post_complete_message_to_sweep(tc_args, config):
    pipe_path = "/tmp/pipe_transformer_qa"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s\n%s" % (str(tc_args), str(config)))


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')

    set_seed(7)

    # arguments
    model_type = args.model_type
    model_name = args.model_name
    qa_args = QuestionAnsweringArgs()
    qa_args.load(args.model_name)
    qa_args.model_name = model_name
    qa_args.model_type = model_type
    qa_args.update_from_dict({"num_train_epochs": args.num_train_epochs,
                              "learning_rate": args.learning_rate,
                              "gradient_accumulation_steps": args.gradient_accumulation_steps,
                              "do_lower_case": args.do_lower_case,
                              "manual_seed": args.manual_seed,
                              "reprocess_input_data": False,
                              "overwrite_output_dir": True,
                              "max_seq_length": args.max_seq_length,
                              "train_batch_size": args.train_batch_size,
                              "eval_batch_size": args.eval_batch_size,
                              "fp16": args.fp16,
                              "n_gpu": args.n_gpu,
                              "data_dir": args.data_dir,
                              "dataset": args.dataset,
                              "output_dir": args.output_dir,
                              "process_count": 1,
                              "is_debug_mode": args.is_debug_mode,
                              "eval_data_path": args.eval_data_file})

    model_config, model, tokenizer = create_model(qa_args)
    tc_data_manager = QADatasetManager(qa_args, args, tokenizer)

    """
        PipeTransformer related
    """
    config = ConfigArgs()
    config.b_auto_dp = args.b_auto_dp
    config.b_freeze = args.b_freeze
    config.b_auto_pipe = args.b_auto_pipe
    config.b_cache = args.b_cache
    config.freeze_strategy = args.freeze_strategy

    config.is_infiniband = args.is_infiniband
    config.master_addr = args.master_addr
    config.master_port = args.master_port
    config.if_name = args.if_name
    config.num_nodes = args.nnodes
    config.node_rank = args.node_rank
    config.local_rank = args.local_rank

    config.pipe_len_at_the_beginning = args.pipe_len_at_the_beginning
    config.num_chunks_of_micro_batches = args.num_chunks_of_micro_batches

    config.learning_task = config.LEARNING_TASK_QUESTION_ANSWERING
    config.model_name = config.MODEL_BERT
    config.num_layer = model_config.num_hidden_layers
    config.hidden_size = model_config.hidden_size
    config.seq_len = qa_args.max_seq_length
    config.batch_size = args.train_batch_size

    config.is_debug_mode = args.is_debug_mode

    pipe_transformer = PipeTransformer(config, tc_data_manager, model_config, model)
    args.global_rank = pipe_transformer.get_global_rank()
    qa_args.update_from_dict({"global_rank": args.global_rank})
    logging.info("successfully create PipeTransformer. args = " + str(args))

    logging.info(args)
    if args.global_rank == 0:
        run = wandb.init(project="pipe_and_ddp",
                         name="PipeTransformer-r" + str(args.run_id) + "-" + str(args.dataset),
                         config=args)

    # Create a ClassificationModel.
    trainer = QuestionAnsweringTrainer(qa_args, tc_data_manager, pipe_transformer)
    trainer.train_model()
    logging.info("finished the training")

    if args.local_rank == 0:
        post_complete_message_to_sweep(qa_args, config)
