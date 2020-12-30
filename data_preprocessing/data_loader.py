import logging
import traceback

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets

from data_preprocessing.imagenet.ImageNet.datasets import ImageNet


class CVDataset:
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.test_sampler = None

        self.args = None
        self.dataset = "cifar10"

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
        # transform_train = transforms.Compose([
        #     # transforms.RandomSizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        #     transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        #     # transforms.Resize(args.img_size),
        #     # transforms.RandomCrop(args.img_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        # ])
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
        del self.train_dataset
        del self.test_dataset
        self.get_data(self.args, self.dataset)
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
        self.train_loader = DataLoader(self.train_dataset,
                                       sampler=self.train_sampler,
                                       batch_size=batch_size,
                                       num_workers=4,
                                       pin_memory=True)

        if self.test_loader is not None:
            del self.test_loader
        self.test_loader = DataLoader(self.test_dataset,
                                      sampler=self.test_sampler,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True)
        return self.train_loader, self.test_loader
