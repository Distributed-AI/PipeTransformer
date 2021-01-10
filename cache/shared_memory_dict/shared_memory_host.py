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

host_memory_percentage = 0.8
disk_memory_percentage = 0.7


def is_disk_storage_full():
    total, used, free = shutil.disk_usage(__file__)
    used_storage_percentage = used / total
    print("is_disk_storage_full. Percentage = " + str(used_storage_percentage))
    return True if used_storage_percentage > disk_memory_percentage else False


def is_host_memory_full():
    memory_cost_percent = 1 - psutil.virtual_memory()[4] / psutil.virtual_memory()[0]
    print("is_host_memory_full. Percentage = " + str(memory_cost_percent))
    return True if memory_cost_percent > host_memory_percentage else False


def _get_or_create_memory_block(
    name: str, size: int
) -> SharedMemory:
    try:
        return SharedMemory(name=name)
    except FileNotFoundError:
        print("FileNotFoundError")
        return SharedMemory(name=name, create=True, size=size)


name = "hidden_feature_"
cnt = 0
for i in range(2000):
    start_loading = time.time()
    hidden_features = torch.randn([500, 197, 768]).numpy()
    cached_hidden_feature = CachedHiddenFeature(i, 1, hidden_features)
    end_loading = time.time()
    print("tensor generation time = " + str(end_loading - start_loading))

    shm_i = _get_or_create_memory_block(name=name + str(i), size=500*197*768*4)

    sharable_hidden_feature = np.ndarray(hidden_features.shape, dtype=hidden_features.dtype, buffer=shm_i.buf)
    sharable_hidden_feature[:] = hidden_features[:]

    shm_cost = time.time() - end_loading
    print("shm_cost time = " + str(shm_cost))

    shm_i.close()
    shm_i.unlink()
    print("---------------------")
    is_disk_storage_full()
    is_host_memory_full()
    cnt += 1
    print(cnt)

    # print(hf.get_layer_id())
    # print(hf.get_np_hidden_feature())
