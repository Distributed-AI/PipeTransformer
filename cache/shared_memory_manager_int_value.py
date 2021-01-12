# https://docs.python.org/3/library/multiprocessing.shared_memory.html
# https://pypi.org/project/shared-memory-dict/
import logging
from multiprocessing.shared_memory import SharedMemory

import numpy
import numpy as np

from cache.shared_memory_dict.lock import lock


class SharedMemoryManagerIntValue:
    def __init__(self, args, name):
        self.args = args
        self.name = name

    @lock
    def set_int_value(self, sample_uid, int_value):
        int_value_np = numpy.asarray(int_value)
        int_name = self._build_layer_id_memory_name(sample_uid)
        size = 4
        tensor_shm = self._get_or_create_memory_block(
            name=int_name,
            size=size
        )
        sharable_hidden_tensor = np.ndarray([1], dtype=numpy.int32,
                                            buffer=tensor_shm.buf)
        sharable_hidden_tensor[0] = int_value_np

    @lock
    def is_exist(self, sample_uid):
        name = self._build_layer_id_memory_name(sample_uid)
        try:
            SharedMemory(name=name)
            return True
        except FileNotFoundError:
            return False

    @lock
    def get_int_value(self, sample_uid):
        int_value = -1
        name = self._build_layer_id_memory_name(sample_uid)
        try:
            shm = SharedMemory(name=name)
        except FileNotFoundError:
            return None
        shm_hidden_tensor_np = np.ndarray(
            [1],
            dtype=numpy.int32,
            buffer=shm.buf
        )
        try:
            int_value = shm_hidden_tensor_np[0]
        except IndexError:
            self._delete_tensor(sample_uid)
            logging.info("global_rank = %d, shm_hidden_tensor_np.size = %d" % (self.args.global_rank, shm_hidden_tensor_np.size))
            raise Exception("_get_tensor error!")
        return int_value

    @lock
    def delete(self, sample_uid):
        self._delete_tensor(sample_uid)

    @lock
    def _delete_tensor(self, sample_uid):
        name = self._build_layer_id_memory_name(sample_uid)
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

    def _build_layer_id_memory_name(self, sample_uid):
        return self.name + "_layer_id_" + str(sample_uid)
