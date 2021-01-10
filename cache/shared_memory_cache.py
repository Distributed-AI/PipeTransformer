# https://docs.python.org/3/library/multiprocessing.shared_memory.html
# https://pypi.org/project/shared-memory-dict/
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch


class SharedMemoryCache:
    def __init__(self, name):
        self.name = name
        self.tensor_shape = None
        self.tensor_dtype = None

    def set(self, sample_uid, layer_id, hidden_tensor):
        hidden_tensor = hidden_tensor.numpy()
        if self.tensor_shape is None:
            self.tensor_shape = hidden_tensor.shape
        if self.tensor_dtype is None:
            self.tensor_dtype = hidden_tensor.dtype
        name = self._build_memory_name(sample_uid, layer_id)
        size = self._get_size_of_tensor(hidden_tensor)
        shm = self._get_or_create_memory_block(
            name=name,
            size=size
        )
        sharable_hidden_tensor = np.ndarray(hidden_tensor.shape, dtype=hidden_tensor.dtype, buffer=shm.buf)
        sharable_hidden_tensor[:] = hidden_tensor[:]
        print(name)

    def get(self, sample_uid, layer_id):
        name = self._build_memory_name(sample_uid, layer_id)
        hidden_tensor_np = np.ndarray(self.tensor_shape, dtype=self.tensor_dtype)
        shm = SharedMemory(name=name)
        hidden_tensor = np.ndarray(self.tensor_shape, dtype=self.tensor_dtype, buffer=shm.buf)
        hidden_tensor_np[:] = hidden_tensor[:]
        tensor = torch.from_numpy(hidden_tensor_np).cpu()
        return tensor

    def delete(self, sample_uid, layer_id):
        name = self._build_memory_name(sample_uid, layer_id)
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            print("%d does not exist" % sample_uid)

    def _get_or_create_memory_block(
            self, name: str, size: int
    ) -> SharedMemory:
        try:
            return SharedMemory(name=name)
        except FileNotFoundError:
            print("create new shared_memory")
            return SharedMemory(name=name, create=True, size=size)

    def _build_memory_name(self, sample_uid, layer_id):
        return self.name + "_" + str(sample_uid) + "_" + str(layer_id)

    def _get_size_of_tensor(self, tensor):
        shape = tensor.shape
        size = 4
        for i in shape:
            size *= i
        return size


if __name__ == "__main__":
    sm_cache = SharedMemoryCache("hidden_feature7")

    blocks = 10
    for i in range(blocks):
        hidden_tensor_set = torch.randn([500, 197, 768])
        sm_cache.set(i, i, hidden_tensor_set)

        hidden_tensor = sm_cache.get(i, i)
        if torch.equal(hidden_tensor, hidden_tensor_set):
            print("equal")

    print("************")
    for i in range(blocks):
        sm_cache.delete(i, i)
