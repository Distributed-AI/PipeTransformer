import math
import pickle
import time

import torch
import torch.multiprocessing as mp


def disk_cache_process_impl(data_queue_list, msg_q):
    window_len = 3
    while True:
        print("disk_process - MyProcess. run()")
        (is_writing, chunk_idx, chunk_num) = msg_q.get()

        print("disk_process - is_writing = " + str(is_writing))
        print("disk_process - chunk_idx = " + str(chunk_idx))
        if is_writing:
            # 1. find the chunk index list which can be cached into the disk storage
            chunk_index_list_to_be_cached = find_the_chunks_to_be_cache(window_len, chunk_idx, chunk_num)

            # 2. cache to disk with pickle file
            save_as_pickle_file(data_queue_list, chunk_index_list_to_be_cached)
        else:
            # 1. find the chunk index list that should be loaded into the host memory,
            # and the chunk list that should be put into the disk storage
            chunk_index_list_to_be_cached, chunk_index_list_to_be_loaded = find_the_chunks_for_reading(window_len, chunk_idx, chunk_num)

            # 2. load from disk storage
            load_from_pickle_file(data_queue_list, chunk_index_list_to_be_loaded)

            # 3. cache to disk with pickle file
            save_as_pickle_file(data_queue_list, chunk_index_list_to_be_cached)


def find_the_chunks_to_be_cache(window_len, current_chunk_index, chunk_num):
    chunk_list_to_be_cache = []
    # keep the first 2 in memory
    if current_chunk_index < window_len:
        return chunk_list_to_be_cache


    return chunk_list_to_be_cache


def find_the_chunks_for_reading(window_len, chunk_idx, chunk_num):
    chunk_index_list_to_be_cached = []
    chunk_index_list_to_be_loaded = []
    return chunk_index_list_to_be_cached, chunk_index_list_to_be_loaded


def save_as_pickle_file(data_queue_list, chunk_index_list_to_be_cached):
    path = ""
    pickle.dump(data_queue_list, open(path, "wb"))


def load_from_pickle_file(data_queue_list, chunk_index_list_to_be_loaded):
    path = ""
    return pickle.load(open(path, "rb"))



class AutoCache:
    def __init__(self):
        self.is_enable = False

        self.num_frozen_layers = 0

        self.batch_size_train = 500
        self.batch_size_test = 100

        self.chunk_num = 10

        self.chunk_idx = -1
        self.chunk_batch_idx = -1

        self.data_q = dict()
        # disk storage process
        for c_i in range(self.chunk_num):
            data_q_i = mp.Queue()
            self.data_q[c_i] = data_q_i

        self.msg_q = mp.Queue()
        self.disk_storage_process = mp.Process(target=disk_cache_process_impl, args=(self.data_q, self.msg_q))
        self.disk_storage_process.daemon = True
        self.disk_storage_process.start()

        is_writing = True
        chunk_size = math.ceil(self.batch_size_train/self.chunk_num)
        print("main process - chunk_size = %d" % chunk_size)
        for batch_idx in range(self.batch_size_train):
            print("main process - main process writing. batch_idx = " + str(batch_idx))

            chunk_idx = int(batch_idx / chunk_size)
            print("main process - chunk_idx = %d" % chunk_idx)

            if self.chunk_idx != chunk_idx:
                self.chunk_idx = chunk_idx
                self.msg_q.put((is_writing, chunk_idx, self.chunk_num))

            hidden_feature = torch.rand([2, 3])
            self.data_q[chunk_idx].put(hidden_feature)

        time.sleep(5)

        is_writing = False
        for batch_idx in range(self.batch_size_train):
            print("main process - main process reading. batch_idx = " + str(batch_idx))

            chunk_idx = int(batch_idx / chunk_size)
            print("main process - chunk_idx = %d" % chunk_idx)

            if self.chunk_idx != chunk_idx:
                self.chunk_idx = chunk_idx
                self.msg_q.put((is_writing, chunk_idx))

            hidden_feature = self.data_q[chunk_idx].get()
            print("main process - " + str(hidden_feature))

        time.sleep(1000)


if __name__ == '__main__':
    auto_cache = AutoCache()
