# https://docs.python.org/3/library/multiprocessing.shared_memory.html
# https://pypi.org/project/shared-memory-dict/
import copy
import shutil
import time
from multiprocessing.shared_memory import SharedMemory
from time import sleep

import numpy as np
import psutil
import torch

from cache.cached_hidden_feature import CachedHiddenFeature
from cache.shared_memory_dict import SharedMemoryDict

hidden_features = torch.randn([500, 197, 768]).numpy()
shape = hidden_features.shape
size = 4
for i in shape:
    size *= i
print(size)