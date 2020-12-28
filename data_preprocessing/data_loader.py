import torch
import torch.distributed as dist
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, DistributedSampler
from torchvision import transforms, datasets

from data_preprocessing.imagenet.ImageNet.datasets import ImageNet


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


def load_imagenet_centralized_training_for_vit(args):
    if args.is_distributed == 1:
        torch.distributed.barrier()

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


def get_data_loader(trainset, testset, batch_size, rank):
    """
    Optimization:
        Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
    """
    train_sampler = DistributedSampler(trainset, rank=rank)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None
    return train_loader, test_loader