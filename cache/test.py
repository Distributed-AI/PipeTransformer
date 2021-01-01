import logging
import math
import pickle
import time

import torch
import torch.multiprocessing as mp
import os

import wandb


def disk_cache_process_impl(data_queue_list, msg_q):
    window_len = 3
    path = "./hidden_feature_cache"
    while True:
        # print("disk_process - MyProcess. run()")
        (is_writing, chunk_idx, chunk_num) = msg_q.get()

        # print("disk_process - is_writing = " + str(is_writing))
        # print("disk_process - chunk_idx = " + str(chunk_idx))
        if is_writing:
            # 1. find the chunk index list which can be cached into the disk storage
            chunk_index_list_to_be_cached = find_the_chunks_to_be_cache(window_len, chunk_idx, chunk_num)
            logging.info("chunk_index_list_to_be_cached = %s" % str(chunk_index_list_to_be_cached))

            # 2. cache to disk with pickle file
            save_as_pickle_file(path, chunk_idx, data_queue_list, chunk_index_list_to_be_cached)
        else:
            # 1. find the chunk index list that should be loaded into the host memory,
            # and the chunk list that should be put into the disk storage
            chunk_index_list_to_be_loaded = find_the_chunks_for_reading(data_queue_list, window_len, chunk_idx, chunk_num)
            chunk_index_list_to_be_cached = find_the_chunks_to_be_cache(window_len, chunk_idx, chunk_num)
            logging.info("chunk_idx = %d, chunk_index_list_to_be_loaded = %s" % (chunk_idx, str(chunk_index_list_to_be_loaded)))
            logging.info("chunk_idx = %d, chunk_index_list_to_be_cached = %s" % (chunk_idx, str(chunk_index_list_to_be_cached)))

            # 2. load from disk storage
            load_from_pickle_file(path, data_queue_list, chunk_index_list_to_be_loaded)

            # 3. cache to disk with pickle file
            save_as_pickle_file(path, chunk_idx, data_queue_list, chunk_index_list_to_be_cached)


def find_the_chunks_to_be_cache(window_len, current_chunk_index, chunk_num):
    chunk_list_to_be_cache = []
    if current_chunk_index < window_len:
        chunk_list_to_be_cache = []
    else:
        for idx in range(window_len, current_chunk_index):
            chunk_list_to_be_cache.append(idx)
    return chunk_list_to_be_cache


def find_the_chunks_for_reading(data_queue_list, window_len, chunk_idx, chunk_num):
    chunk_index_list_to_be_loaded = []
    for idx in range(window_len):
        loaded_idx = chunk_idx + idx + 1
        if loaded_idx >= chunk_num:
            continue
        is_in_the_memory = True
        if data_queue_list[loaded_idx].empty():
            is_in_the_memory = False
        if not is_in_the_memory and loaded_idx < chunk_num:
            chunk_index_list_to_be_loaded.append(loaded_idx)
    return chunk_index_list_to_be_loaded


def move_tensor_queue_to_numpy_dict(tensor_queue):
    numpy_dict = dict()
    idx = 0
    while not tensor_queue.empty():
        tensor = tensor_queue.get()
        numpy_dict[idx] = tensor.numpy()
        idx += 1
    return numpy_dict


def move_numpy_dict_to_tensor_queue(data_queue_list, chunk_idx, numpy_dict):
    for idx in range(len(numpy_dict.keys())):
        numpy_feature = numpy_dict[idx]
        # print("numpy_feature = " + str(numpy_feature))
        tensor = torch.from_numpy(numpy_feature)
        data_queue_list[chunk_idx].put(tensor)


def save_as_pickle_file(path, chunk_idx, data_queue_list, chunk_index_list_to_be_cached):
    logging.info("chunk_idx = %d, chunk_index_list_to_be_cached = %s" % (chunk_idx, str(chunk_index_list_to_be_cached)))
    for chunk_idx in chunk_index_list_to_be_cached:
        if not data_queue_list[chunk_idx].empty():
            numpy_queue = move_tensor_queue_to_numpy_dict(data_queue_list[chunk_idx])
            pickle.dump(numpy_queue, open(path + str(chunk_idx) + ".fea", "wb"))
            logging.info("----------chunk_idx = %d save successfully----------" % chunk_idx)


def load_from_pickle_file(path, data_queue_list, chunk_index_list_to_be_loaded):
    logging.info("chunk_index_list_to_be_loaded = " + str(chunk_index_list_to_be_loaded))
    for chunk_idx in chunk_index_list_to_be_loaded:
        loading_path = path + str(chunk_idx) + ".fea"
        data_dict = pickle.load(open(loading_path, "rb"))
        logging.info("--------chunk_idx = %d, data_dict.size() = %d" % (chunk_idx, len(data_dict)))
        move_numpy_dict_to_tensor_queue(data_queue_list, chunk_idx, data_dict)
        os.remove(loading_path)
        logging.info("----------chunk_idx = %d load successfully----------" % (chunk_idx))


class AutoCache:
    def __init__(self):
        self.is_enable = False

        self.num_frozen_layers = 0

        self.batch_size_train = 80
        self.batch_size_test = 10

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
        chunk_size = math.ceil(self.batch_size_train / self.chunk_num)
        logging.info("main process - chunk_size = %d" % chunk_size)
        for batch_idx in range(self.batch_size_train):
            logging.info("main process - main process writing. batch_idx = " + str(batch_idx))

            chunk_idx = math.floor(batch_idx / chunk_size)
            logging.info("main process - chunk_idx = %d" % chunk_idx)

            if self.chunk_idx != chunk_idx:
                self.chunk_idx = chunk_idx
                self.msg_q.put((is_writing, chunk_idx, self.chunk_num))

            hidden_feature = torch.rand([1000, 7680, 197])
            self.data_q[chunk_idx].put(hidden_feature)
            # ime.sleep(0.1)

        time.sleep(5)

        logging.info("---------------------\n")
        logging.info("---------------------\n")
        logging.info("---------------------\n")
        is_writing = False
        for batch_idx in range(self.batch_size_train):
            logging.info("main process - main process reading. batch_idx = " + str(batch_idx))

            chunk_idx = math.floor(batch_idx / chunk_size)
            logging.info("main process - chunk_idx = %d" % chunk_idx)

            if self.chunk_idx != chunk_idx:
                self.chunk_idx = chunk_idx
                self.msg_q.put((is_writing, chunk_idx, self.chunk_num))

            if self.data_q[chunk_idx].empty():
                logging.info("empty")
            hidden_feature = self.data_q[chunk_idx].get()
            logging.info("main process - " + str(hidden_feature))
            # time.sleep(0.1)

        time.sleep(1000)


if __name__ == '__main__':
    wandb.init(project="pipe_and_ddp",
                     name="PipeTransformer-Cache Test")
    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(processName)s - %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')

    auto_cache = AutoCache()

