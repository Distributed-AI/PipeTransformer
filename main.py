import argparse
import logging
import os
import socket

import numpy as np
import psutil
import wandb

from cache.auto_cache import AutoCache
from data_preprocessing.cv_data_manager import CVDatasetManager
from dp.auto_dp import AutoDataParallel
from freeze.auto_freeze import AutoFreeze
from model.vit.vision_transformer_origin import CONFIGS
from model.vit.vision_transformer_origin import VisionTransformer
from pipe.auto_pipe import AutoElasticPipe
from pipe.pipe_model_builder import OutputHead
from trainer import VisionTransformerTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PipeTransformer: Elastic and Automated Pipelining for Fast Distributed Training of Transformer Models")
    parser.add_argument("--nnodes", type=int, default=2)

    parser.add_argument("--nproc_per_node", type=int, default=8)

    parser.add_argument("--node_rank", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--global_rank", type=int, default=0)

    parser.add_argument("--master_addr", type=str, default="192.168.11.2")

    parser.add_argument("--master_port", type=int, default=22222)

    parser.add_argument("--if_name", type=str, default="lo")

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
                        default="./model/vit/pretrained/ViT-B_16.npz",
                        help="Where to search for pretrained vit models.")

    parser.add_argument("--is_distributed", default=1, type=int,
                        help="is_distributed")

    parser.add_argument("--pipe_len_at_the_beginning", default=4, type=int,
                        help="pipe_len_at_the_beginning")

    parser.add_argument("--is_infiniband", default=1, type=int,
                        help="is_infiniband")

    # pipelin = 2 * 8
    # pipelin = 2 * 4
    # pipelin = 2 * 2
    parser.add_argument("--num_chunks_of_micro_batches", default=16, type=int,
                        help="num_chunks_of_micro_batches")

    parser.add_argument('--freeze', dest='b_freeze', action='store_true')
    parser.add_argument('--no_freeze', dest='b_freeze', action='store_false')
    parser.set_defaults(b_freeze=True)

    parser.add_argument('--auto_pipe', dest='b_auto_pipe', action='store_true')
    parser.add_argument('--do_auto_pipe', dest='b_auto_pipe', action='store_false')
    parser.set_defaults(b_auto_pipe=True)

    parser.add_argument('--auto_dp', dest='b_auto_dp', action='store_true')
    parser.add_argument('--no_auto_dp', dest='b_auto_dp', action='store_false')
    parser.set_defaults(b_auto_dp=False)

    parser.add_argument('--cache', dest='b_cache', action='store_true')
    parser.add_argument('--no_cache', dest='b_cache', action='store_false')
    parser.set_defaults(b_cache=True)

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.ERROR,
                        format='%(processName)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    hostname = socket.gethostname()
    logging.info("#############process ID = " +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # init ddp global group
    auto_dp = AutoDataParallel(args)
    auto_dp.init_ddp(args)
    auto_dp.init_rpc()
    auto_dp.enable(args.b_auto_dp)
    args.global_rank = auto_dp.get_global_rank()

    if args.global_rank == 0:
        run = wandb.init(project="pipe_and_ddp",
                         name="PipeTransformer""-" + str(args.dataset),
                         config=args)

    # create dataset
    cv_data_manager = CVDatasetManager(args)
    output_dim = cv_data_manager.get_output_dim()

    # create model
    model_type = 'vit-B_16'
    # model_type = 'vit-L_32'
    # model_type = 'vit-H_14'
    config = CONFIGS[model_type]
    args.num_layer = config.transformer.num_layers
    args.transformer_hidden_size = config.hidden_size
    args.seq_len = 197

    logging.info("Vision Transformer Configuration: " + str(config))
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=output_dim, vis=False)
    model.load_from(np.load(args.pretrained_dir))
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info("model_size = " + str(model_size))

    output_head = OutputHead(config.hidden_size, output_dim)

    num_layers = config.transformer.num_layers
    logging.info("num_layers = %d" % num_layers)

    # create AutoFreeze algorithm
    auto_freeze = AutoFreeze(args)
    auto_freeze.enable(args.b_freeze)

    # create pipe and DDP
    auto_pipe = AutoElasticPipe(auto_dp.get_world_size(), args.local_rank, args.global_rank, args.num_chunks_of_micro_batches,
                                model, output_head, args.pipe_len_at_the_beginning, num_layers)
    auto_pipe.enable(args.b_auto_pipe)

    # create FP cache with CPU memory
    auto_cache = AutoCache(args, auto_freeze, auto_dp, auto_pipe, cv_data_manager, model.get_hidden_feature_size() * args.batch_size)
    auto_cache.enable(args.b_cache)
    # start training
    freeze_point = dict()
    freeze_point['epoch'] = 0
    frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed = auto_dp.transform(auto_pipe, auto_freeze,
                                                                                               None, model,
                                                                                               0, freeze_point)
    freeze_point = auto_dp.get_freeze_point()

    trainer = VisionTransformerTrainer(args, auto_freeze, auto_pipe, auto_dp, auto_cache,
                                       frozen_model, pipe_model, cv_data_manager)
    trainer.train_and_eval(freeze_point)

    auto_cache.cleanup()
    auto_freeze.cleanup()

    if args.global_rank == 0:
        wandb.finish()
