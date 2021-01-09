from torchvision import transforms, datasets

import logging
import os
import random

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets

from data_preprocessing.imagenet.ImageNet.datasets import ImageNet
from utils import *


class CVDatasetManager:
    def __init__(self, num_train_epochs):
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.test_sampler = None

        self.args = None
        self.dataset = "cifar10"

        self.num_train_epochs = num_train_epochs
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

        num_workers = 4
        """
        for imagenet, we need to reduce the memory cost:
        https://github.com/prlz77/ResNeXt.pytorch/issues/5
        """
        if self.dataset == "imagenet":
            num_workers = 1
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

        num_workers = 4
        """
        for imagenet, we need to reduce the memory cost:
        https://github.com/prlz77/ResNeXt.pytorch/issues/5
        """
        if self.dataset == "imagenet":
            num_workers = 1
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

    def create_mapping_for_single_worker_training(self):
        for i in range(int(self.num_train_epochs)):
            self.set_seed(self.seeds[i])
            tmp_indices = [i*10 for i in range(self.num_train_samples)]
            random_to_original = {}
            train_indices_dataloader = DataLoader(tmp_indices, sampler=RandomSampler(tmp_indices), num_workers=0)
            for step, batch in enumerate(train_indices_dataloader):
                random_to_original[step] = batch[0].item()
            self.origin_sample_id_mapping_by_epoch.append(random_to_original)

    def create_mapping_for_distributed_training(self):
        for i in range(int(self.num_train_epochs)):
            self.set_seed(self.seeds[i])
            tmp_indices = [i*10 for i in range(self.num_train_samples)]
            random_to_original = {}
            train_indices_dataloader = DataLoader(tmp_indices, sampler=RandomSampler(tmp_indices), num_workers=0)
            for step, batch in enumerate(train_indices_dataloader):
                random_to_original[step] = batch[0].item()
            self.origin_sample_id_mapping_by_epoch.append(random_to_original)

    def get_seed_by_epoch(self, epoch):
        return self.seeds[epoch]

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    data_manager = CVDatasetManager(2)
    data_manager.create_mapping()
    print(data_manager.origin_sample_id_mapping_by_epoch[0])

