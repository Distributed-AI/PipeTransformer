import argparse
import logging
import os
import socket
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import wandb
from torch.utils.data import SequentialSampler, DataLoader, DistributedSampler
from torchvision import transforms, datasets

from dp.auto_dp import AutoDataParallel
from freeze.auto_freeze import AutoFreeze
from model.vit.vision_transformer_origin import VisionTransformer
from model.vit.vision_transformer_task_specific_layer import CONFIGS
from pipe.auto_pipe import AutoElasticPipe
from pipe.pipe_model_builder import OutputHead
from utils import WarmupCosineSchedule, WarmupLinearSchedule


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def load_cifar_centralized_training_for_vit(args):
    # if args.is_distributed == 1:
    #     torch.distributed.barrier()

    """
        the std 0.5 normalization is proposed by BiT (Big Transfer), which can increase the accuracy around 3%
    """
    # CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    # CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    CIFAR_MEAN = [0.5, 0.5, 0.5]
    CIFAR_STD = [0.5, 0.5, 0.5]

    """
        transforms.RandomSizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)) leads to a very low training accuracy.
    """
    transform_train = transforms.Compose([
        # transforms.RandomSizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.Resize(args.img_size),
        transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform_test)
        output_dim = 10
    else:
        trainset = datasets.CIFAR100(root=args.data_dir,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir,
                                    train=False,
                                    download=True,
                                    transform=transform_test)
        output_dim = 100

    # if args.is_distributed == 1:
    #     torch.distributed.barrier()
    return trainset, testset, output_dim


def get_data_loader(trainset, testset, rank):
    train_sampler = DistributedSampler(trainset, rank=rank)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None
    return train_loader, test_loader


def train(args, auto_pipe, auto_dp, model, epoch, train_dataloader, test_dataloader):
    if auto_freeze.is_freeze_open():
        new_freeze_point = dict()
        new_freeze_point['epoch'] = epoch
        model = auto_dp.transform(auto_pipe, model, auto_freeze.get_hand_crafted_frozen_layers_by_epoch(epoch), new_freeze_point)
        new_freeze_point = auto_dp.get_freeze_point()
        new_train_dl, new_test_dl = get_data_loader(train_dataset, test_dataset, auto_dp.get_data_rank())
        train_dataloader = new_train_dl
    else:
        new_train_dl = train_dataloader
        new_test_dl = test_dataloader

    num_sample_processed_in_total = 0
    communication_count = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer(model)

    global size_params_ddp_sum
    device_first = auto_pipe.get_device_first()
    device_last = auto_pipe.get_device_last()
    print("device_first = " + str(device_first))
    print("device_last = " + str(device_last))

    def sync_all_devices(local_rank, device_cnt=4):
        for d in range(device_cnt):
            device = torch.device("cuda:" + str(local_rank + d))
            torch.cuda.synchronize(device)

    # measure latency with cuda event:
    # https://discuss.pytorch.org/t/distributed-training-slower-than-dataparallel/81539/4
    model.train()
    # with torch.cuda.device(device_first):
    start_ld = torch.cuda.Event(enable_timing=True)
    end_ld = torch.cuda.Event(enable_timing=True)
    start_fp = torch.cuda.Event(enable_timing=True)
    end_bp = torch.cuda.Event(enable_timing=True)

    # with torch.cuda.device(device_last):
    end_fp = torch.cuda.Event(enable_timing=True)
    start_bp = torch.cuda.Event(enable_timing=True)

    iteration_num = 0
    for batch_idx, (x, target) in enumerate(train_dataloader):
        if batch_idx == 0:
            starting_time = time.time()
        logging.info("--------------Epoch %d, batch index %d Statistics: " % (epoch, batch_idx))
        logging.info("global_rank = %d. epoch = %d, batch index = %d/%d" % (auto_dp.get_global_rank(), epoch, batch_idx, len(train_dl)))
        num_sample_processed_in_total += len(x)
        communication_count += 1
        iteration_num += 1

        # load data
        # with torch.cuda.device(device_first):
        start_ld.record()
        x = x.to(device_first)
        target = target.to(device_last)

        # with torch.cuda.device(device_first):
        end_ld.record()

        # FP
        # with torch.cuda.device(device_first):
        start_fp.record()

        optimizer.zero_grad()
        log_probs = model(x)

        # with torch.cuda.device(device_last):
        end_fp.record()

        # BP
        # with torch.cuda.device(device_last):
        start_bp.record()

        loss = criterion(log_probs, target)
        loss.backward()
        # this clip will cost 0.6 second, can be skipped?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # with torch.cuda.device(device_first):
        end_bp.record()

        # recv_gbyte, transmit_gbyte = net_meter.update_bandwidth()
        # logging.info("BW {recv_MB:%.3f} {transmit_MB:%.3f}" % (recv_gbyte * 1024, transmit_gbyte * 1024))


        sync_all_devices(0, auto_pipe.get_pipe_len())
        if batch_idx == 0:
            time_finish_prepare_ddp = time.time()
            logging.info("data loading cost = " + str(time_finish_prepare_ddp - starting_time))

        # with torch.cuda.device(device_first):
        logging.info(f"data loading time cost (s) by CUDA event {start_ld.elapsed_time(end_ld)/1000}")
        # with torch.cuda.device(device_first):
        logging.info(f"forward time cost (s) by CUDA event {start_fp.elapsed_time(end_fp)/1000}")
        # with torch.cuda.device(device_last):
        logging.info(f"backwards time cost: (s) by CUDA event {start_bp.elapsed_time(end_bp)/1000}")


        sample_num_throughput = int(
            num_sample_processed_in_total / (time.time() - time_finish_prepare_ddp)) * auto_dp.get_active_world_size()
        logging.info("global_rank = %d. sample_num_throughput (images/second): %d" % (auto_dp.get_global_rank(),
                                                                                      sample_num_throughput))

        comm_freq = communication_count / (time.time() - time_finish_prepare_ddp)
        logging.info("global_rank = %d. communication frequency (cross machine sync/second): %f" % (auto_dp.get_global_rank(),
                                                                                                    comm_freq))

        logging.info("global_rank = %d. size_params_ddp_sum: %f" % (auto_dp.get_global_rank(),
                                                                    size_params_ddp_sum / 1e6))
        logging.info("-------------------------------------")
        size_params_ddp_sum = 0.0

        if iteration_num == 2 and args.is_debug_mode:
            break
    return model, device_first, device_last, new_train_dl, new_test_dl


def _infer(model, test_data, device_first, device_last):
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss()
    iteration_num = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            logging.info("evaluation - batch index = %d/%d" % (batch_idx, len(test_data)))
            iteration_num += 1
            x = x.to(device_first)
            target = target.to(device_last)
            log_probs = model(x)
            loss = criterion(log_probs, target)
            _, predicted = torch.max(log_probs, -1)
            correct = predicted.eq(target).sum()
            test_acc += correct.item()
            test_loss += loss.item() * target.size(0)
            test_total += target.size(0)
            if iteration_num == 2 and args.is_debug_mode:
                break

    return test_acc, test_total, test_loss


def eval(model, args, epoch, train_dl, test_dl, device_first, device_last):
    # train data
    if epoch == args.epochs - 1:
        train_tot_correct, train_num_sample, train_loss = _infer(model, train_dl, device_first, device_last)
        # test on training dataset
        train_acc = train_tot_correct / train_num_sample
        train_loss = train_loss / train_num_sample

        if args.global_rank == 0:
            wandb.log({"Train/Acc": train_acc, "epoch": epoch})
            wandb.log({"Train/Loss": train_loss, "epoch": epoch})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

    # test data
    # if (epoch + 1) % args.freq_eval_test_acc == 0:
    if epoch == args.epochs - 1:
        test_tot_correct, test_num_sample, test_loss = _infer(model, test_dl, device_first, device_last)

        # test on test dataset
        test_acc = test_tot_correct / test_num_sample
        test_loss = test_loss / test_num_sample

        if args.global_rank == 0:
            wandb.log({"Test/Acc": test_acc, "epoch": epoch})
            wandb.log({"Test/Loss": test_loss, "epoch": epoch})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)


def train_and_eval(auto_pipe, auto_dp, model, train_dl, test_dl, freeze_point, args):
    epoch_start = freeze_point['epoch']
    for epoch in range(epoch_start, args.epochs):
        model, device_first, device_last, new_train_dl, new_test_dl = train(args, auto_pipe, auto_dp, model, epoch, train_dl, test_dl)
        eval(model, args, epoch, new_train_dl, new_test_dl, device_first, device_last)


def build_optimizer(model):
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
    return optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP Demo")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--global_rank", type=int, default=0)

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

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    args = parser.parse_args()
    print(args)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # global variables
    size_params_ddp_sum = 0.0

    # init ddp global group
    auto_dp = AutoDataParallel(args.pipe_len_at_the_beginning)
    auto_dp.init_ddp(args)
    auto_dp.init_rpc()
    auto_dp.warm_up()
    args.global_rank = auto_dp.get_global_rank()

    if args.global_rank == 0:
        wandb.init(project="pipe_and_ddp",
                   name="pipe_and_ddp(c)" + str(args.epochs) + "-lr" + str(args.lr),
                   config=args)

    # create dataset
    train_dataset, test_dataset, output_dim = load_cifar_centralized_training_for_vit(args)

    # create model
    model_type = 'vit-B_16'
    # model_type = 'vit-L_32'
    # model_type = 'vit-H_14'
    config = CONFIGS[model_type]

    print("Vision Transformer Configuration: " + str(config))
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=output_dim, vis=False)
    model.load_from(np.load(args.pretrained_dir))
    model_size = count_parameters(model)
    print("model_size = " + str(model_size))

    output_head = OutputHead(config.hidden_size, output_dim)

    num_layers = config.transformer.num_layers
    print("num_layers = %d" % num_layers)

    # create AutoFreeze algorithm
    auto_freeze = AutoFreeze()
    # auto_freeze.do_not_freeze()

    # create pipe and DDP
    auto_pipe = AutoElasticPipe(auto_dp.get_world_size(), args.local_rank, args.global_rank, model,
                                output_head, args.pipe_len_at_the_beginning, num_layers)

    # start training
    freeze_point = dict()
    freeze_point['epoch'] = 0
    model = auto_dp.transform(auto_pipe, model, 0, freeze_point)
    freeze_point = auto_dp.get_freeze_point()
    train_dl, test_dl = get_data_loader(train_dataset, test_dataset, auto_dp.get_data_rank())
    train_and_eval(auto_pipe, auto_dp, model, train_dl, test_dl, freeze_point, args)
