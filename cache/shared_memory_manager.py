# https://docs.python.org/3/library/multiprocessing.shared_memory.html
# https://pypi.org/project/shared-memory-dict/
import logging
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch


class SharedMemoryManager:
    def __init__(self, name):
        self.name = name
        self.tensor_shape = None
        self.tensor_dtype = None

        self.layer_id_shape = None
        self.layer_id_dtype = None

    def set(self, sample_uid, layer_id, hidden_tensor):
        self._set_tensor(sample_uid, hidden_tensor)
        self._set_layer_id(sample_uid, layer_id)

    def _set_tensor(self, sample_uid, hidden_tensor):
        hidden_tensor = hidden_tensor.numpy()
        if self.tensor_shape is None:
            self.tensor_shape = hidden_tensor.shape
        if self.tensor_dtype is None:
            self.tensor_dtype = hidden_tensor.dtype
        tensor_name = self._build_tensor_memory_name(sample_uid)
        tensor_size = self._get_size_of_tensor(hidden_tensor)
        tensor_shm = self._get_or_create_memory_block(
            name=tensor_name,
            size=tensor_size
        )
        sharable_hidden_tensor = np.ndarray(hidden_tensor.shape, dtype=hidden_tensor.dtype, buffer=tensor_shm.buf)
        sharable_hidden_tensor[:] = hidden_tensor[:]

    def _set_layer_id(self, sample_uid, layer_id):
        layer_id_np = np.array([layer_id])
        if self.layer_id_shape is None:
            self.layer_id_shape = layer_id_np.shape
        if self.layer_id_dtype is None:
            self.layer_id_dtype = layer_id_np.dtype
        layer_id_name = self._build_layer_id_memory_name(sample_uid)
        layer_id_size = 8
        layer_id_shm = self._get_or_create_memory_block(
            name=layer_id_name,
            size=layer_id_size
        )
        sharable_layer_id = np.ndarray(layer_id_np.shape, dtype=layer_id_np.dtype, buffer=layer_id_shm.buf)
        sharable_layer_id[:] = layer_id_np[:]

    def get(self, sample_uid, tensor_np, idx_in_batch):
        return self._get_tensor(sample_uid, tensor_np, idx_in_batch), self._get_layer_id(sample_uid)

    def _get_tensor(self, sample_uid, tensor_np, idx_in_batch):
        name = self._build_tensor_memory_name(sample_uid)
        try:
            shm = SharedMemory(name=name)
        except FileNotFoundError:
            return None
        hidden_tensor = np.ndarray(self.tensor_shape, dtype=self.tensor_dtype, buffer=shm.buf)
        try:
            tensor_np[idx_in_batch] = hidden_tensor[:]
        except IndexError:
            self._delete_tensor(sample_uid)
            logging.info(hidden_tensor.size)
            logging.info("_get_tensor error!")
            return None
        return tensor_np

    def _get_layer_id(self, sample_uid):
        layer_id_name = self._build_layer_id_memory_name(sample_uid)
        try:
            shm = SharedMemory(name=layer_id_name)
        except FileNotFoundError:
            return None
        layer_id_np = np.ndarray(self.layer_id_shape, dtype=self.layer_id_dtype)
        layer_id_shm = np.ndarray(self.layer_id_shape, dtype=self.layer_id_dtype, buffer=shm.buf)
        try:
            layer_id_np[:] = layer_id_shm[:]
        except IndexError:
            self._delete_layer_id(sample_uid)
            logging.info(layer_id_np.size)
            logging.info(layer_id_shm.size)
            logging.info("_get_layer_id error!")
            return None
        return layer_id_np[0]

    def delete(self, sample_uid, layer_id):
        self._delete_tensor(sample_uid)
        self._delete_layer_id(layer_id)

    def _delete_tensor(self, sample_uid):
        name = self._build_tensor_memory_name(sample_uid)
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logging.info("%d does not exist" % sample_uid)

    def _delete_layer_id(self, sample_uid):
        name = self._build_layer_id_memory_name(sample_uid)
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logging.info("%d does not exist" % sample_uid)

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

    def _build_layer_id_memory_name(self, sample_uid):
        return self.name + "_layer_" + str(sample_uid)

    def _get_size_of_tensor(self, tensor):
        return tensor.size*4


if __name__ == "__main__":
    sm_cache = SharedMemoryManager("hidden_feature7")

    blocks = 10
    hidden_tensor, layer_id = sm_cache.get(0)
    for i in range(blocks):
        hidden_tensor_set = torch.randn([500, 197, 768])
        sm_cache.set(i, i, hidden_tensor_set)

        hidden_tensor, layer_id = sm_cache.get(i)
        print("layer_id = %d" % layer_id)
        if torch.equal(hidden_tensor, hidden_tensor_set):
            print("equal")

    print("************")
    for i in range(blocks):
        sm_cache.delete(i, i)
