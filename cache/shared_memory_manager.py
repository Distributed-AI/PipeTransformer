# https://docs.python.org/3/library/multiprocessing.shared_memory.html
# https://pypi.org/project/shared-memory-dict/
import copy
import logging
from multiprocessing.shared_memory import SharedMemory

import numpy
import numpy as np
import torch

from cache.shared_memory_dict.lock import lock


class SharedMemoryManager:
    def __init__(self, args, name):
        self.args = args
        self.name = name

    @lock
    def set_tensor(self, sample_uid, tensor):
        shm_hidden_tensor_np = tensor.numpy()
        tensor_name = self._build_tensor_memory_name(sample_uid)
        tensor_size = 4 * self.args.seq_len * self.args.transformer_hidden_size
        tensor_shm = self._get_or_create_memory_block(
            name=tensor_name,
            size=tensor_size
        )
        sharable_hidden_tensor = np.ndarray([self.args.seq_len, self.args.transformer_hidden_size], dtype=shm_hidden_tensor_np.dtype,
                                            buffer=tensor_shm.buf)
        sharable_hidden_tensor[:] = shm_hidden_tensor_np[:]

    @lock
    def is_exist(self, sample_uid):
        name = self._build_tensor_memory_name(sample_uid)
        try:
            SharedMemory(name=name)
            return True
        except FileNotFoundError:
            return False

    @lock
    def get_tensor(self, sample_uid, dtype):
        name = self._build_tensor_memory_name(sample_uid)
        try:
            shm = SharedMemory(name=name)
        except FileNotFoundError:
            raise Exception("get_tensor not found")
        shm_hidden_tensor_np = np.ndarray(
            [self.args.seq_len, self.args.transformer_hidden_size],
            dtype=dtype,
            buffer=shm.buf
        )
        return torch.from_numpy(copy.deepcopy(shm_hidden_tensor_np[:]))

    @lock
    def delete(self, sample_uid):
        self._delete_tensor(sample_uid)

    @lock
    def _delete_tensor(self, sample_uid):
        name = self._build_tensor_memory_name(sample_uid)
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logging.info("%d does not exist" % sample_uid)

    @lock
    def _get_or_create_memory_block(
            self, name: str, size: int
    ) -> SharedMemory:
        try:
            return SharedMemory(name=name)
        except FileNotFoundError:
            # logging.info("create new shared_memory")
            return SharedMemory(name=name, create=True, size=size)

    def _build_tensor_memory_name(self, sample_uid):
        return self.name + "_tensor_" + str(sample_uid)
