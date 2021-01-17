# https://docs.python.org/3/library/multiprocessing.shared_memory.html
# https://pypi.org/project/shared-memory-dict/
import logging
from multiprocessing.shared_memory import SharedMemory

import numpy
import numpy as np

from .lock import lock


class SharedMemoryManagerIntValue:
    def __init__(self, args, name):
        self.args = args
        self.name = name
        self.non_shared_memory_for_cleanup_layer_id = dict()

    @lock
    def add_int_value(self, sample_uid, int_value):
        int_value_np = numpy.asarray(int_value)
        int_name = self._build_layer_id_memory_name(sample_uid)
        tensor_shm = SharedMemory(name=int_name, create=True, size=4)
        sharable_hidden_tensor = np.ndarray([1], dtype=numpy.int32,
                                            buffer=tensor_shm.buf)
        sharable_hidden_tensor[0] = int_value_np
        self.non_shared_memory_for_cleanup_layer_id[sample_uid] = int_name

    @lock
    def set_int_value(self, sample_uid, int_value):
        int_value_np = numpy.asarray(int_value)
        int_name = self._build_layer_id_memory_name(sample_uid)
        tensor_shm = SharedMemory(name=int_name)
        sharable_hidden_tensor = np.ndarray([1], dtype=numpy.int32,
                                            buffer=tensor_shm.buf)
        sharable_hidden_tensor[0] = int_value_np
        self.non_shared_memory_for_cleanup_layer_id[sample_uid] = int_name

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
            self.delete_tensor(sample_uid)
            logging.info(
                "global_rank = %d, shm_hidden_tensor_np.size = %d" % (self.args.global_rank, shm_hidden_tensor_np.size))
            raise Exception("_get_tensor error!")
        return int_value

    @lock
    def delete_tensor(self, sample_uid):
        name = self._build_layer_id_memory_name(sample_uid)
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logging.info("%d does not exist" % sample_uid)

    @lock
    def cleanup(self):
        for sample_uid in self.non_shared_memory_for_cleanup_layer_id.keys():
            self.delete_tensor(sample_uid)

    def _build_layer_id_memory_name(self, sample_uid):
        return self.name + "_layer_id_" + str(sample_uid)
