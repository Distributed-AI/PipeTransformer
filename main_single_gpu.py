import argparse
import logging
import os
import socket
import time

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, DistributedSampler
from torchvision import transforms, datasets

from meter import NetworkMeter
from model.vit.vision_transformer_origin import VisionTransformer
from model.vit.vision_transformer_task_specific_layer import CONFIGS
from utils import WarmupCosineSchedule, WarmupLinearSchedule


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def load_cifar_centralized_training_for_vit(args, global_rank=0):
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

    train_sampler = RandomSampler(trainset) if args.is_distributed == 0 else DistributedSampler(trainset,
                                                                                                rank=global_rank)
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

    return train_loader, test_loader, output_dim


def train_cpu(model, epoch, train_dl, criterion, optimizer, scheduler, device_first, device_last):
    model.train()
    num_sample_processed_in_total = 0

    for batch_idx, (x, target) in enumerate(train_dl):
        logging.info("epoch = %d, batch index = %d/%d" % (epoch, batch_idx, len(train_dl)))
        num_sample_processed_in_total += len(x)

        # load data
        time_load_data = time.time()
        x = x.to(device_first)
        target = target.to(device_last)
        time_end_load = time.time()
        logging.info("  load time cost: %f" % (time_end_load - time_load_data))

        # FP
        time_start_fp = time.time()

        optimizer.zero_grad()
        log_probs = model(x)

        time_end_fp = time.time()
        logging.info("  fp time cost: %f" % (time_end_fp - time_start_fp))

        # BP
        time_start_bp = time.time()

        loss = criterion(log_probs, target)
        loss.backward()
        # this clip will cost 0.6 second, can be skipped?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        time_end_bp = time.time()
        logging.info("  bp time cost: %f" % (time_end_bp - time_start_bp))


def train(model, epoch, train_dl, criterion, optimizer, scheduler, device_first, device_last):
    model.train()
    num_sample_processed_in_total = 0

    start_ld = torch.cuda.Event(enable_timing=True)
    end_ld = torch.cuda.Event(enable_timing=True)
    start_fp = torch.cuda.Event(enable_timing=True)
    end_fp = torch.cuda.Event(enable_timing=True)
    start_bp = torch.cuda.Event(enable_timing=True)
    end_bp = torch.cuda.Event(enable_timing=True)
    for batch_idx, (x, target) in enumerate(train_dl):
        logging.info("epoch = %d, batch index = %d/%d" % (epoch, batch_idx, len(train_dl)))
        num_sample_processed_in_total += len(x)

        with torch.cuda.device(device_first):
            first_partition_stream = torch.cuda.current_stream()

        with torch.cuda.device(device_last):
            last_partition_stream = torch.cuda.current_stream()

        # load data
        start_ld.record(first_partition_stream)
        x = x.to(device_first)
        target = target.to(device_last)
        end_ld.record(first_partition_stream)

        # FP
        start_fp.record(first_partition_stream)

        optimizer.zero_grad()
        log_probs = model(x)

        end_fp.record(last_partition_stream)

        # BP
        start_bp.record(last_partition_stream)

        loss = criterion(log_probs, target)
        loss.backward()
        # this clip will cost 0.6 second, can be skipped?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        end_bp.record(first_partition_stream)

        def sync_all_devices(local_rank, device_cnt=4):
            for d in range(device_cnt):
                torch.cuda.synchronize(local_rank+d)

        sync_all_devices(args.local_rank, 1)

        # recv_gbyte, transmit_gbyte = net_meter.update_bandwidth()
        # logging.info("BW {recv_MB:%.3f} {transmit_MB:%.3f}" % (recv_gbyte * 1024, transmit_gbyte * 1024))

        logging.info(f"data loading time cost (ms) by CUDA event {start_ld.elapsed_time(end_ld)}")
        logging.info(f"forward time cost (ms) by CUDA event {start_fp.elapsed_time(end_fp)}")
        logging.info(f"backwards time cost: (ms) by CUDA event {start_bp.elapsed_time(end_bp)}")



def _infer(test_data, device_first, device_last):
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            logging.info("evaluation - batch index = %d/%d" % (batch_idx, len(train_dl)))
            x = x.to(device_first)
            target = target.to(device_last)
            log_probs = model(x)
            loss = criterion(log_probs, target)
            _, predicted = torch.max(log_probs, -1)
            correct = predicted.eq(target).sum()
            test_acc += correct.item()
            test_loss += loss.item() * target.size(0)
            test_total += target.size(0)

    return test_acc, test_total, test_loss


def eval(args, epoch, train_dl, test_dl, device_first, device_last):
    # train data
    if (epoch + 1) % args.freq_eval_train_acc == 0:
        train_tot_correct, train_num_sample, train_loss = _infer(train_dl, device_first, device_last)
        # test on training dataset
        train_acc = train_tot_correct / train_num_sample
        train_loss = train_loss / train_num_sample

        if args.global_rank == 0:
            wandb.log({"Train/Acc": train_acc, "epoch": epoch})
            wandb.log({"Train/Loss": train_loss, "epoch": epoch})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

    # test data
    if (epoch + 1) % args.freq_eval_test_acc == 0:
        test_tot_correct, test_num_sample, test_loss = _infer(test_dl, device_first, device_last)

        # test on test dataset
        test_acc = test_tot_correct / test_num_sample
        test_loss = test_loss / test_num_sample

        if args.global_rank == 0:
            wandb.log({"Test/Acc": test_acc, "epoch": epoch})
            wandb.log({"Test/Loss": test_loss, "epoch": epoch})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)


def train_and_eval(model, train_dl, test_dl, args, device_first, device_last):
    criterion = nn.CrossEntropyLoss().to(device)
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

    for epoch in range(args.epochs):
        if args.is_gpu:
            train(model, epoch, train_dl, criterion, optimizer, scheduler, device_first, device_last)
        else:
            train_cpu(model, epoch, train_dl, criterion, optimizer, scheduler, device_first, device_last)
        eval(args, epoch, train_dl, test_dl, device_first, device_last)


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

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
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

    parser.add_argument('--epochs', type=int, default=20, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument("--freq_eval_train_acc", default=4, type=int)

    parser.add_argument("--freq_eval_test_acc", default=1, type=int)

    parser.add_argument("--is_gpu", default=1, type=int)

    parser.add_argument("--is_distributed", default=0, type=int,
                        help="is_distributed")

    parser.add_argument("--pretrained_dir", type=str,
                        default="./model/vit/pretrained/ViT-B_16.npz",
                        help="Where to search for pretrained vit models.")

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

    wandb.init(project="pipe_and_ddp",
               name="ddp-" + str(args.epochs) + "-lr" + str(args.lr),
               config=args)

    # create dataset
    train_dl, test_dl, output_dim = load_cifar_centralized_training_for_vit(args)
    if args.is_gpu:
        device = torch.device("cuda:" + str(args.local_rank))
    else:
        device = torch.device("cpu")

    # create model
    img_size = args.img_size
    model_type = 'vit-B_16'
    pretrained_dir = ""

    # pretrained on ImageNet (224x224), and fine-tuned on (384x384) high resolution.
    config = CONFIGS[model_type]
    logging.info("Vision Transformer Configuration: " + str(config))
    model = VisionTransformer(config, img_size, zero_head=True, num_classes=output_dim, vis=False)
    model.load_from(np.load(args.pretrained_dir))
    model.to(device)
    model_size = count_parameters(model)
    logging.info("model_size = " + str(model_size))

    print(model)
    # model size
    num_params = count_parameters(model)
    logging.info("Vision Transformer Model Size = " + str(num_params))

    # start training
    train_and_eval(model, train_dl, test_dl, args, device, device)
