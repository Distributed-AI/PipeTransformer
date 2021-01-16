import logging
import math
import os
import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []

    data_local_num_dict = dict()
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        target_num = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    target_num += 1

        net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        data_local_num_dict[class_to_idx[target]] = target_num
        sum_temp += target_num

    assert len(images) == sum_temp
    return images, data_local_num_dict, net_dataidx_map


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageNet(data.Dataset):

    def __init__(self, data_dir, batch_size, node_num=0, nproc_per_node=0, node_rank=-1,
                 dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        """
            Generating this class too many times will be time-consuming.
            So it will be better calling this once and put it into ImageNet_truncated.
        """
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.loader = default_loader
        if self.train:
            self.data_dir = os.path.join(data_dir, 'train')
        else:
            self.data_dir = os.path.join(data_dir, 'val')

        self.all_data, self.data_local_num_dict, self.net_dataidx_map = self.__getdatasets__()
        if dataidxs == None:
            self.local_data = self.all_data
        elif type(dataidxs) == int:
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin: end]
        else:
            self.local_data = []
            for idxs in dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]

        # for PipeTransformer
        if node_num > 0 and node_rank >= 0:
            data_len = len(self.local_data)
            if data_len % node_num > 0:
                subset_len = math.ceil(data_len / node_num)
                even_len = subset_len * node_num
                self.local_data += self.local_data[:even_len - data_len]
                starting_idx = subset_len * node_rank
                end_idx = subset_len * (node_rank + 1)
                # logging.info("data_len = %d, node_num = %d" % (len(self.local_data), node_num))
                # raise Exception("dataset cannot be partitioned to equal length!")
            else:
                subset_len = int(data_len / node_num)
                starting_idx = subset_len * node_rank
                end_idx = subset_len * (node_rank + 1)
            self.local_data = self.local_data[starting_idx:end_idx]

        # drop_last = False
        data_len = len(self.local_data)
        if data_len % nproc_per_node > 0:
            subset_len = math.ceil(data_len / nproc_per_node)
            even_len = int(subset_len * nproc_per_node)
            temp = self.local_data[:even_len - data_len]
            self.local_data = np.concatenate((self.local_data, temp), axis=0)

        data_len = len(self.local_data)
        gap = data_len % batch_size
        if gap > 0:
            added_batch = batch_size - gap
            temp = self.local_data[:added_batch]
            self.local_data = np.concatenate((self.local_data, temp), axis=0)
        logging.info("data_len = %d" % len(self.local_data))

    def get_local_data(self):
        return self.local_data

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def __getdatasets__(self):
        # all_data = datasets.ImageFolder(data_dir, self.transform, self.target_transform)

        classes, class_to_idx = find_classes(self.data_dir)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        all_data, data_local_num_dict, net_dataidx_map = make_dataset(self.data_dir, class_to_idx, IMG_EXTENSIONS)
        if len(all_data) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_dir + "\n"
                                                                                     "Supported extensions are: " + ",".join(
                extensions)))
        return all_data, data_local_num_dict, net_dataidx_map

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        return len(self.local_data)
