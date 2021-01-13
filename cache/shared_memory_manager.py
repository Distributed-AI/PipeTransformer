# https://docs.python.org/3/library/multiprocessing.shared_memory.html
# https://pypi.org/project/shared-memory-dict/
import copy
import logging
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

from cache.shared_memory_dict.lock import lock


class SharedMemoryManager:
    def __init__(self, args, name):
        self.args = args
        self.name = name
        self.non_shared_memory_for_cleanup_tensor = dict()

    @lock
    def add_tensor(self, sample_uid, layer_id, tensor):
        shm_hidden_tensor_np = tensor.numpy()
        tensor_name = self._build_tensor_memory_name(sample_uid, layer_id)
        tensor_size = shm_hidden_tensor_np.nbytes
        try:
            tensor_shm = SharedMemory(name=tensor_name, create=True, size=tensor_size)
            sharable_hidden_tensor = np.ndarray(shape=shm_hidden_tensor_np.shape, dtype=shm_hidden_tensor_np.dtype,
                                                buffer=tensor_shm.buf)
            np.copyto(sharable_hidden_tensor, shm_hidden_tensor_np)

            self.non_shared_memory_for_cleanup_tensor[sample_uid] = tensor_name
        except FileExistsError:
            logging.info("%s is already stored!" % tensor_name)
            raise FileExistsError
    @lock
    def update_tensor(self, sample_uid, layer_id, tensor):
        shm_hidden_tensor_np = tensor.numpy()
        tensor_name = self._build_tensor_memory_name(sample_uid, layer_id)
        tensor_shm = SharedMemory(name=tensor_name)
        sharable_hidden_tensor = np.ndarray(shape=shm_hidden_tensor_np.shape, dtype=shm_hidden_tensor_np.dtype,
                                            buffer=tensor_shm.buf)
        np.copyto(sharable_hidden_tensor, shm_hidden_tensor_np)

    @lock
    def get_tensor(self, sample_uid, layer_id):
        name = self._build_tensor_memory_name(sample_uid, layer_id)
        try:
            shm = SharedMemory(name=name)
        except FileNotFoundError:
            # raise Exception("get_tensor not found")
            return None
        if len(shm.buf) != self.args.seq_len * self.args.transformer_hidden_size * 4:
            logging.info("len(shm.buf) = %d" % len(shm.buf))
            raise Exception("length is incorrect!")
        shm_hidden_tensor_np = np.ndarray(
            shape=(self.args.seq_len, self.args.transformer_hidden_size),
            dtype=np.float32,
            buffer=shm.buf
        )
        tensor = torch.from_numpy(copy.deepcopy(shm_hidden_tensor_np[:]))
        del shm_hidden_tensor_np
        shm.close()
        return tensor

    @lock
    def delete_tensor(self, sample_uid, layer_id):
        name = self._build_tensor_memory_name(sample_uid, layer_id)
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logging.info("%d does not exist" % sample_uid)

    def delete_tensor_by_name(self, name):
        try:
            logging.critical("deleting %s" % name)
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logging.info("%d does not exist" % name)

    @lock
    def cleanup(self):
        for sample_uid in self.non_shared_memory_for_cleanup_tensor.keys():
            name = self.non_shared_memory_for_cleanup_tensor[sample_uid]
            self.delete_tensor_by_name(name)

    def _build_tensor_memory_name(self, sample_uid, layer_id):
        return self.name + "_tensor_" + str(layer_id) + "_" + str(sample_uid)
