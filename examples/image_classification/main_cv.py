import argparse
import logging
import os
import socket
import sys

import numpy as np
import psutil
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from pipe_transformer.config_args import ConfigArgs
from pipe_transformer.pipe_transformer import PipeTransformer

from pipe_transformer.data.cv_data_manager import CVDatasetManager
from model.cv.vision_transformer_origin import CONFIGS
from model.cv.vision_transformer_origin import VisionTransformer

from examples.image_classification.cv_trainer import CVTrainer


def add_args():
    parser = argparse.ArgumentParser(
        description="PipeTransformer: "
                    "Elastic and Automated Pipelining for Fast Distributed Training of Transformer Models")

    # PipeTransformer related
    parser.add_argument("--nnodes", type=int, default=2)

    parser.add_argument("--nproc_per_node", type=int, default=8)

    parser.add_argument("--node_rank", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--global_rank", type=int, default=0)

    parser.add_argument("--master_addr", type=str, default="192.168.11.2")

    parser.add_argument("--master_port", type=int, default=22222)

    parser.add_argument("--if_name", type=str, default="lo")

    parser.add_argument("--is_distributed", default=1, type=int,
                        help="is_distributed")

    parser.add_argument("--pipe_len_at_the_beginning", default=4, type=int,
                        help="pipe_len_at_the_beginning")

    parser.add_argument("--is_infiniband", default=1, type=int,
                        help="is_infiniband")

    parser.add_argument("--num_chunks_of_micro_batches", default=1 * 8, type=int,
                        help="num_chunks_of_micro_batches")

    parser.add_argument("--freeze_strategy", type=str, default="mild")

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
    parser.set_defaults(b_cache=False)


    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # model related
    parser.add_argument('--model', type=str, default='transformer', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./data/cifar10',
                        help='data directory')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")

    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.03)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0)

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")

    parser.add_argument("--warmup_steps", default=2, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument("--freq_eval_train_acc", default=4, type=int)

    parser.add_argument("--freq_eval_test_acc", default=1, type=int)

    parser.add_argument("--pretrained_dir", type=str,
                        default="./../../model/cv/pretrained/ViT-B_16.npz",
                        help="Where to search for pretrained vit models.")

    args = parser.parse_args()
    return args


def post_complete_message_to_sweep(args, config):
    pipe_path = "/tmp/pipe_transformer_training_status_cv"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s\n%s" % (str(args), str(config)))


if __name__ == "__main__":
    args = add_args()

    # customize the log format
    logging.basicConfig(level=logging.DEBUG,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    hostname = socket.gethostname()
    logging.info("#############process ID = " +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    """
        Dataset related
    """
    cv_data_manager = CVDatasetManager(args)
    cv_data_manager.set_seed(0)
    output_dim = cv_data_manager.get_output_dim()

    """
        Model related
    """
    model_type = 'vit-B_16'
    # model_type = 'vit-L_32'
    # model_type = 'vit-H_14'
    model_config = CONFIGS[model_type]
    model_config.output_dim = output_dim
    args.num_layer = model_config.transformer.num_layers
    args.transformer_hidden_size = model_config.hidden_size
    args.seq_len = 197

    logging.info("Vision Transformer Configuration: " + str(model_config))
    model = VisionTransformer(model_config, args.img_size, zero_head=True, num_classes=output_dim, vis=False)
    model.load_from(np.load(args.pretrained_dir))
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info("model_size = " + str(model_size))

    num_layers = model_config.transformer.num_layers
    logging.info("num_layers = %d" % num_layers)

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

    config.learning_task = config.LEARNING_TASK_IMAGE_CLASSIFICATION
    config.model_name = config.MODEL_VIT
    config.num_layer = num_layers
    config.output_dim = output_dim
    config.hidden_size = args.transformer_hidden_size
    config.seq_len = args.seq_len
    config.batch_size = args.batch_size

    config.is_debug_mode = args.is_debug_mode

    pipe_transformer = PipeTransformer(config, cv_data_manager, model_config, model)
    args.global_rank = pipe_transformer.get_global_rank()

    """
        Logging related
    """
    if args.global_rank == 0:
        run = wandb.init(project="pipe_and_ddp",
                         name="PipeTransformer""-" + str(args.dataset),
                         config=args)

    """
        Trainer related
    """
    trainer = CVTrainer(args, pipe_transformer)
    trainer.train_and_eval()

    """
        PipeTransformer related
    """
    pipe_transformer.finish()

    if args.global_rank == 0:
        wandb.finish()

    if args.local_rank == 0:
        post_complete_message_to_sweep(args, config)
