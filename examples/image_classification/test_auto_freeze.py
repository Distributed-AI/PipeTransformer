import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from examples.image_classification.utils import WarmupCosineSchedule, WarmupLinearSchedule
from model.cv.vision_transformer_origin import CONFIGS, VisionTransformer
from pipe_transformer.data.cv_data_manager import CVDatasetManager
from pipe_transformer.freeze.auto_freeze import AutoFreeze
from pipe_transformer.pipe.pipe_model_builder import OutputHead

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PipeTransformer: Elastic and Automated Pipelining for Fast Distributed Training of Transformer Models")
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

    parser.add_argument('--batch_size', type=int, default=6, metavar='N',
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

    parser.add_argument("--do_freeze", default=1, type=int,
                        help="do freeze")

    parser.add_argument("--do_cache", default=0, type=int,
                        help="do cache")

    parser.add_argument("--is_debug_mode", default=1, type=int,
                        help="is_debug_mode")

    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    # create dataset
    cv_data_manager = CVDatasetManager()
    train_dataset, test_dataset, output_dim = cv_data_manager.get_data(args, args.dataset)

    # create model
    model_type = 'vit-B_16'
    # model_type = 'vit-L_32'
    # model_type = 'vit-H_14'
    config = CONFIGS[model_type]

    logging.info("Vision Transformer Configuration: " + str(config))
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=output_dim, vis=False)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logging.info("model_size = " + str(model_size))

    output_head = OutputHead(config.hidden_size, output_dim)

    num_layers = config.transformer.num_layers
    logging.info("num_layers = %d" % num_layers)

    # create AutoFreeze algorithm
    auto_freeze = AutoFreeze()
    if args.do_freeze == 0:
        auto_freeze.do_not_freeze()

    train_dl, test_dl = cv_data_manager.get_data_loader(args.batch_size, 2, 0)

    criterion = torch.nn.CrossEntropyLoss()
    if args.client_optimizer == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr,
                                     weight_decay=args.wd, amsgrad=True)

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=args.epochs)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=args.epochs)
    if args.global_rank == 0:
        model.train()
        model.to(0)
        iteration_num = 0
        freq_freeze_check = 10
        for batch_idx, (x, target) in enumerate(train_dl):
            iteration_num += 1

            x = x.to(0)
            target = target.to(0)

            optimizer.zero_grad()
            log_probs = model(x)

            loss = criterion(log_probs, target)
            loss.backward()
            # this clip will cost 0.6 second, can be skipped?
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            auto_freeze.accumulate(model)

            if iteration_num % freq_freeze_check == 0:
                auto_freeze.freeze(model)

            if iteration_num == 100 and args.is_debug_mode:
                break
