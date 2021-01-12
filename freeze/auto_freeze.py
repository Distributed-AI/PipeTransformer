import argparse
import copy
import logging

import numpy as np
import torch

from cache.shared_memory_dict.shared_memory_dict import SharedMemoryDict
from cache.shared_memory_manager import SharedMemoryManager
from data_preprocessing.cv_data_manager import CVDatasetManager
from model.vit.vision_transformer_origin import CONFIGS, VisionTransformer
from pipe.pipe_model_builder import OutputHead
from utils import WarmupCosineSchedule, WarmupLinearSchedule


class AutoFreeze:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.num_freeze_layers = 0
        self.is_freeze = False
        self.is_hand_crafted = True

        self.is_grad_norm_analysis = False

        self.num_layer = 12
        self.grad_accumulated_by_layer = dict()
        for layer_idx in range(self.num_layer):
            self.grad_accumulated_by_layer[layer_idx] = dict()
        self.is_grad_accumulated_by_layer_updated = False

        self.freeze_interval = 1

        self.last_grad_norm_by_layer = None
        self.percentile = 50

        self.shared_memory_dict_frozen_layer_num = SharedMemoryDict("frozen_layer_num", 4)

    def update_status(self, num_freeze_layers, last_grad_norm_by_layer):
        logging.info("(%s) num_freeze_layers = %d, last_grad_norm_by_layer = %s" % (str(id(self)), num_freeze_layers, str(last_grad_norm_by_layer)))
        self.num_freeze_layers = num_freeze_layers
        if last_grad_norm_by_layer is not None:
            self.last_grad_norm_by_layer = copy.deepcopy(last_grad_norm_by_layer)

    def get_status(self):
        return self.num_freeze_layers, self.last_grad_norm_by_layer

    def enable(self, on):
        self.is_freeze = on

    def is_freeze_open(self):
        return self.is_freeze

    def get_hand_crafted_frozen_layers_by_epoch(self, epoch):
        num_freeze_layers = 0
        if epoch == 0:
            num_freeze_layers = 0
        elif epoch >= 1 and epoch <= 2:
            num_freeze_layers = 6
        elif epoch > 2 and epoch <= 5:
            num_freeze_layers = 8
        elif epoch > 5 and epoch <= 7:
            num_freeze_layers = 10
        elif epoch > 7:
            num_freeze_layers = 12
        self.shared_memory_dict_frozen_layer_num[epoch] = num_freeze_layers
        return num_freeze_layers

    def get_num_of_frozen_layer(self, epoch):
        return self.shared_memory_dict_frozen_layer_num[epoch]

    def accumulate(self, model):
        for layer_idx in range(self.num_layer):
            for name, param in model.transformer.encoder.layer[layer_idx].named_parameters():
                if param.grad is not None:
                    if name not in self.grad_accumulated_by_layer.keys():
                        self.grad_accumulated_by_layer[layer_idx][name] = param.grad
                    else:
                        self.grad_accumulated_by_layer[layer_idx][name] += param.grad
                    # logging.info("layer_idx = %d, name = %s" % (layer_idx, name))
        self.is_grad_accumulated_by_layer_updated = True

    def freeze(self, epoch):
        logging.info("-----------------------------%s" % (id(self)))
        if self.is_hand_crafted:
            return self.get_hand_crafted_frozen_layers_by_epoch(epoch)
        if epoch == 0:
            return 0

        if not self.is_grad_accumulated_by_layer_updated:
            return self.num_freeze_layers
        if self.num_freeze_layers == self.num_layer:
            return self.num_freeze_layers

        if (epoch + 1) % self.freeze_interval == 0:
            # Calculate layer-wise gradient changing ratio
            grad_norm_by_layer = dict()
            for i in range(self.num_layer):
                grad_norm_by_layer[i] = 0

            for layer_idx in self.grad_accumulated_by_layer.keys():
                for name in self.grad_accumulated_by_layer[layer_idx].keys():
                    grad = self.grad_accumulated_by_layer[layer_idx][name]
                    grad_norm_by_layer[layer_idx] += torch.norm(grad.cpu().detach(), p=1).item()

            # Clear gradient accumulator
            for layer_idx in self.grad_accumulated_by_layer.keys():
                for name in self.grad_accumulated_by_layer[layer_idx].keys():
                    params = self.grad_accumulated_by_layer[layer_idx][name]
                    self.grad_accumulated_by_layer[layer_idx][name] = torch.zeros(params.shape)
            self.is_grad_accumulated_by_layer_updated = False

            logging.info("epoch = %d, grad_norm_by_layer = %s" % (epoch, str(grad_norm_by_layer)))
            frozen_layer_idx = -1
            if self.last_grad_norm_by_layer is None:
                # Set gradient dict to be compared with for the first time
                self.last_grad_norm_by_layer = grad_norm_by_layer
            else:
                grad_norm_diff = dict()
                # Calculate gradient changing threshold
                for key in grad_norm_by_layer.keys():
                    if grad_norm_by_layer[key] > 0:
                        grad_norm_diff[key] = abs(self.last_grad_norm_by_layer[key] - grad_norm_by_layer[key]) / \
                                              self.last_grad_norm_by_layer[key]
                    else:
                        grad_norm_diff[key] = 0

                logging.info(grad_norm_diff)
                unfrozen_list = list(grad_norm_diff.values())[self.num_freeze_layers:]
                logging.info(unfrozen_list)
                logging.info("epoch = %d, grad_norm_diff (unfrozen_list) = %s" % (epoch, str(unfrozen_list)))
                grad_norm_diff_percentile = np.percentile(unfrozen_list, self.percentile)
                logging.info("grad_norm_diff_percentile = " + str(grad_norm_diff_percentile))

                # Find out the first layer with ratio ge to the median value
                for layer_idx in grad_norm_diff.keys():
                    if grad_norm_diff[layer_idx] >= grad_norm_diff_percentile:
                        frozen_layer_idx = layer_idx
                        break

                self.last_grad_norm_by_layer = grad_norm_by_layer
            logging.info("epoch = %d, frozen_layer_idx = %s" % (epoch, str(frozen_layer_idx)))
            # only analyze the grad norm
            if self.is_grad_norm_analysis:
                return 0
            if frozen_layer_idx != -1:
                self.num_freeze_layers = frozen_layer_idx + 1
                self.shared_memory_dict_frozen_layer_num[epoch] = self.num_freeze_layers
        logging.info("epoch = %d, num_frozen_layer = %s" % (epoch, str(self.num_freeze_layers)))
        return self.num_freeze_layers


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
