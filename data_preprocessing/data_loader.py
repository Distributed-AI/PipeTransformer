import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from data_preprocessing.cifar.cifar_dataset import CIFAR10, CIFAR100
from data_preprocessing.imagenet.imagenet_datasets import ImageNet


class CVDatasetManager:
    def __init__(self, args):
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.test_sampler = None

        self.args = args
        self.dataset = args.dataset

        # only load dataset once
        self.get_data(args, self.dataset)

        self.num_train_epochs = args.epochs
        self.num_train_samples = 20
        self.origin_sample_id_mapping_by_epoch = []
        self.seeds = [i for i in range(self.num_train_epochs)]

    def get_data(self, args, dataset):
        self.args = args
        self.dataset = dataset
        logging.info("load_data. dataset_name = %s" % dataset)
        if dataset == "cifar10":
            train_dataset, test_dataset, output_dim = self.load_cifar_centralized_training_for_vit(args)
        elif dataset == "cifar100":
            train_dataset, test_dataset, output_dim = self.load_cifar_centralized_training_for_vit(args)
        elif dataset == "imagenet":
            train_dataset, test_dataset, output_dim = self.load_imagenet_centralized_training_for_vit(args)
        else:
            raise Exception("no such dataset!")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        return train_dataset, test_dataset, output_dim

    def load_cifar_centralized_training_for_vit(self, args):
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
            # transforms.RandomCrop(args.img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        if args.dataset == "cifar10":
            trainset = CIFAR10(root=args.data_dir,
                               train=True,
                               download=True,
                               transform=transform_train)
            testset = CIFAR10(root=args.data_dir,
                              train=False,
                              download=True,
                              transform=transform_test)
            output_dim = 10
        else:
            trainset = CIFAR100(root=args.data_dir,
                                train=True,
                                download=True,
                                transform=transform_train)
            testset = CIFAR100(root=args.data_dir,
                               train=False,
                               download=True,
                               transform=transform_test)
            output_dim = 100

        # if args.is_distributed == 1:
        #     torch.distributed.barrier()
        return trainset, testset, output_dim

    def load_imagenet_centralized_training_for_vit(self, args):
        # if args.is_distributed == 1:
        #     torch.distributed.barrier()

        """
        the std 0.5 normalization is proposed by BiT (Big Transfer), which can increase the accuracy around 3%
        """
        CIFAR_MEAN = [0.5, 0.5, 0.5]
        CIFAR_STD = [0.5, 0.5, 0.5]

        """
        transforms.RandomSizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)) leads to a very low training accuracy.
        transforms.RandomSizedCrop() is deprecated.
        The following two transforms are equal.
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

        trainset = ImageNet(data_dir=args.data_dir,
                            train=True,
                            download=True,
                            transform=transform_train)
        testset = ImageNet(data_dir=args.data_dir,
                           train=False,
                           download=True,
                           transform=transform_test)
        output_dim = 1000

        return trainset, testset, output_dim

    def get_data_loader(self, batch_size, num_replicas, rank):
        # traceback.print_stack()
        logging.info("---num_replicas = %d, rank = %d --------------" % (num_replicas, rank))
        # del self.train_dataset
        # del self.test_dataset
        # self.get_data(self.args, self.dataset)
        """
        Optimization:
            Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
        """
        if self.train_sampler is not None:
            del self.train_sampler
        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=num_replicas, rank=rank)
        #
        # test_sampler = SequentialSampler(testset)

        if self.test_sampler is not None:
            del self.test_sampler
        self.test_sampler = DistributedSampler(self.test_dataset, num_replicas=num_replicas, rank=rank)

        if self.train_loader is not None:
            del self.train_loader

        num_workers = 0
        """
        for imagenet, we need to reduce the memory cost:
        https://github.com/prlz77/ResNeXt.pytorch/issues/5
        """
        self.train_loader = DataLoader(self.train_dataset,
                                       sampler=self.train_sampler,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=True)

        if self.test_loader is not None:
            del self.test_loader
        self.test_loader = DataLoader(self.test_dataset,
                                      sampler=self.test_sampler,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      pin_memory=True)
        return self.train_loader, self.test_loader

    def get_data_loader_single_worker(self, batch_size):
        """
        Optimization:
            Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
        """
        if self.train_sampler is not None:
            del self.train_sampler
        self.train_sampler = RandomSampler(self.train_dataset)
        #
        # test_sampler = SequentialSampler(testset)

        if self.test_sampler is not None:
            del self.test_sampler
        self.test_sampler = RandomSampler(self.test_dataset)

        # if self.train_loader is not None:
        #     del self.train_loader

        num_workers = 0
        """
        for imagenet, we need to reduce the memory cost:
        https://github.com/prlz77/ResNeXt.pytorch/issues/5
        """
        self.train_loader = DataLoader(self.train_dataset,
                                       sampler=self.train_sampler,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=True)

        # if self.test_loader is not None:
        #     del self.test_loader
        self.test_loader = DataLoader(self.test_dataset,
                                      sampler=self.test_sampler,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      pin_memory=True)
        return self.train_loader, self.test_loader

    def get_seed_by_epoch(self, epoch):
        return self.seeds[epoch]

    def set_seed(self, seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


def test_single_worker():
    args.epochs = 10
    args.img_size = 224
    args.dataset = "cifar10"
    args.data_dir = "./data/cifar10"
    data_manager = CVDatasetManager(args)

    data_manager.set_seed(data_manager.seeds[0])
    batch_size = 8

    # train_indices_dataloader = data_manager.test_mapping_for_single_worker()
    train_indices_dataloader, _ = data_manager.get_data_loader_single_worker(batch_size=batch_size)
    sample_origin_list = {}
    for batch_idx, batch in enumerate(train_indices_dataloader):
        sample_uid_list, sample_origin, target_origin = batch
        sample_uid_list = sample_uid_list.cpu().detach().numpy()
        # logging.info("sample_uid_list = %s" % str(sample_uid_list))
        for sample_i in range(len(sample_uid_list)):
            # logging.info("sample_origin.shape = %s" % str(sample_origin.shape))
            sample_origin_i = sample_origin[sample_i]
            sample_uid = sample_uid_list[sample_i]
            sample_origin_list[sample_uid] = sample_origin_i
    logging.info("sample_origin_list.keys() = %s" + str(sample_origin_list.keys()))
    logging.info("train_indices_dataloader.len = %s" % str(len(train_indices_dataloader)))

    for epoch in range(data_manager.num_train_epochs):
        if epoch == data_manager.num_train_epochs -1:
            break
        num_replicas = 8
        data_manager.set_seed(data_manager.seeds[epoch+1])
        train_loader, test_loader = data_manager.get_data_loader_single_worker(batch_size=batch_size)
        # train_loader = data_manager.test_mapping_for_single_worker()
        logging.info("train_loader.len = %s" % str(len(train_loader)))

        for batch_idx, batch in enumerate(train_loader):
            sample_uid_list, sample, target = batch
            logging.info("---------------------------batch_idx = %d, sample_uid_list = %s" % (batch_idx, str(sample_uid_list)))
            sample_uid_list = sample_uid_list.cpu().detach().numpy()
            for sample_i in range(len(sample)):
                logging.info("sample.len = %s" % str(len(sample)))
                sample_batch_random = sample[sample_i]
                logging.info("sample_batch_random.shape = " + str(sample_batch_random.shape))
                logging.info("sample_uid_list = " + str(sample_uid_list))
                sample_uid = sample_uid_list[sample_i]
                sample_origin = sample_origin_list[sample_uid]
                logging.info("sample_origin.shape = " + str(sample_origin.shape))
                b_equal = torch.equal(sample_batch_random, sample_origin)
                logging.info("b_equal = " + str(b_equal))
                if not b_equal:
                    raise Exception("not equal!")


def test_distributed():
    args.epochs = 10
    args.img_size = 224
    args.dataset = "cifar10"
    args.data_dir = "./data/cifar10"
    data_manager = CVDatasetManager(args)

    data_manager.set_seed(data_manager.seeds[0])
    batch_size = 8
    num_replicas = [2, 2, 2, 2, 4, 4, 8, 8, 16, 16]
    rank = 0

    # train_indices_dataloader = data_manager.test_mapping_for_single_worker()
    train_indices_dataloader, _ = data_manager.get_data_loader(batch_size=batch_size, num_replicas=num_replicas[0], rank=rank)
    # train_indices_dataloader, _ = data_manager.get_data_loader_single_worker(batch_size=batch_size)
    sample_origin_list = {}
    for batch_idx, batch in enumerate(train_indices_dataloader):
        sample_uid_list, sample_origin, target_origin = batch
        sample_uid_list = sample_uid_list.cpu().detach().numpy()
        # logging.info("sample_uid_list = %s" % str(sample_uid_list))
        for sample_i in range(len(sample_uid_list)):
            # logging.info("sample_origin.shape = %s" % str(sample_origin.shape))
            sample_origin_i = sample_origin[sample_i]
            sample_uid = sample_uid_list[sample_i]
            sample_origin_list[sample_uid] = sample_origin_i
    logging.info("sample_origin_list.keys() = %s" + str(sample_origin_list.keys()))
    logging.info("train_indices_dataloader.len = %s" % str(len(train_indices_dataloader)))

    for epoch in range(1, data_manager.num_train_epochs):

        data_manager.set_seed(data_manager.seeds[epoch])
        train_loader, test_loader = data_manager.get_data_loader(batch_size=batch_size, num_replicas=num_replicas[epoch], rank=rank)
        # train_loader = data_manager.test_mapping_for_single_worker()
        logging.info("train_loader.len = %s" % str(len(train_loader)))

        for batch_idx, batch in enumerate(train_loader):
            sample_uid_list, sample, target = batch
            logging.info("---------------------------batch_idx = %d, sample_uid_list = %s" % (batch_idx, str(sample_uid_list)))
            sample_uid_list = sample_uid_list.cpu().detach().numpy()
            for sample_i in range(len(sample)):
                logging.info("sample.len = %s" % str(len(sample)))
                sample_batch_random = sample[sample_i]
                logging.info("sample_batch_random.shape = " + str(sample_batch_random.shape))
                logging.info("sample_uid_list = " + str(sample_uid_list))
                sample_uid = sample_uid_list[sample_i]
                sample_origin = sample_origin_list[sample_uid]
                logging.info("sample_origin.shape = " + str(sample_origin.shape))
                b_equal = torch.equal(sample_batch_random, sample_origin)
                logging.info("b_equal = " + str(b_equal))
                if not b_equal:
                    raise Exception("not equal!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PipeTransformer: Elastic and Automated Pipelining for Fast Distributed Training of Transformer Models")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(processName)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    test_distributed()

    logging.info("done.")
    # logging.info(data_manager.origin_sample_id_mapping_by_epoch[0])
