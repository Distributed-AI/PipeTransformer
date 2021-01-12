import logging
import os
import os.path
import pickle
from os import path

import torch
import torch.multiprocessing as mp

from cache.cache_msg import Message
from cache.disk_memory_manager import DiskMemoryManager
from cache.shared_memory_manager import SharedMemoryManager


class CacheDaemon(mp.Process):
    def __init__(self, args, msg_q):
        super().__init__()
        self.msg_q = msg_q
        self.shared_memory_mgr = SharedMemoryManager(args, "hidden_feature")
        self.disk_memory_mgr = DiskMemoryManager("hidden_feature")

        self.epoch = 0
        self.train_sample_index = []
        self.test_sample_index = []

    def run(self) -> None:
        while True:
            message = self.msg_q.get()
            msg_type = message.get_type()
            if msg_type == Message.MSG_TYPE_TRAINING_PROGRESS:
                logging.info("Message.MSG_TYPE_TRAINING_PROGRESS")
                epoch = message.get(Message.MSG_KEY_EPOCH)
                batch_idx = message.get(Message.MSG_KEY_BATCH_INDEX)
                sample_index_list_to_disk, \
                sample_index_list_to_memory = self._determine_sample_location_with_slding_window(epoch, batch_idx)
                self._move_shared_memory_to_disk(sample_index_list_to_disk)
                self._move_disk_memory_to_shared_memory(sample_index_list_to_memory)

            elif msg_type == Message.MSG_TYPE_UPDATE_INDEX:
                logging.info("Message.MSG_TYPE_UPDATE_INDEX")
                self.epoch = message.get(Message.MSG_KEY_EPOCH)
                self.train_sample_index = message.get(Message.MSG_KEY_TRAIN_SAMPLE_INDEX)
                self.test_sample_index = message.get(Message.MSG_KEY_TRAIN_SAMPLE_INDEX)
                # logging.info(self.train_sample_index)
                # logging.info(self.test_sample_index)

            elif msg_type == Message.MSG_TYPE_RESET:
                logging.info("Message.MSG_TYPE_RESET")
                self._delete_all_cache()
            else:
                raise Exception("no such message")
            logging.info("subprocess is running")

    def _determine_sample_location_with_slding_window(self, epoch, current_batch_idx):
        sample_index_list_to_disk = []
        sample_index_list_to_memory = []
        return sample_index_list_to_disk, sample_index_list_to_memory

    def _calculate_num_of_sample_in_shared_memory(self, available_host_memory, hidden_feature_size):
        pass

    def _calculate_num_of_sample_in_disk_memory(self, available_disk_memory, hidden_feature_size):
        pass

    def _move_shared_memory_to_disk(self, sample_index_list_to_disk):
        pass

    def _move_disk_memory_to_shared_memory(self, sample_index_list_to_memory):
        pass

    def _delete_all_cache(self):
        pass


    def disk_cache_process_impl(self, msg_q):
        host_memory_window_len = -1
        chunk_idx_starting_disk_cache = -1
        chunk_idx_starting_recompute = -1
        chunk_idx_end_cache = -1
        chunk_num = -1
        while True:
            (msg_id, msg_params) = msg_q.get()
            if chunk_num == -1:
                chunk_num = msg_params['chunk_num']
            if msg_id == Message.MSG_TYPE_WRITING:
                logging.info("AutoCache.MSG_TYPE_WRITING")
                chunk_idx = msg_params['chunk_idx']
                logging.info("chunk_idx_starting_disk_cache = %d" % chunk_idx_starting_disk_cache)
                logging.info("chunk_idx_starting_recompute = %d" % chunk_idx_starting_recompute)
                if chunk_idx_starting_disk_cache != -1 and chunk_idx_starting_recompute == -1:
                    chunk_index_list_to_be_cached = []
                    for idx in range(chunk_idx_starting_disk_cache, chunk_idx):
                        chunk_index_list_to_be_cached.append(idx)
                    logging.info("chunk_idx_starting_recompute = %d, chunk_index_list_to_be_cached = %s" % (
                        chunk_idx_starting_recompute, str(chunk_index_list_to_be_cached)))
                    # cache to disk with pickle file
                    self.save_as_pickle_file(cache_path, chunk_size, chunk_idx, data_dict, chunk_index_list_to_be_cached)
                elif chunk_idx_starting_recompute != -1:
                    logging.info("disk memory is full, do nothing")
                else:
                    logging.info("using host memory, do nothing")

            elif msg_id == Message.MSG_TYPE_WRITING_HOST_MEMORY_FULL:
                if chunk_idx_starting_disk_cache == -1:
                    logging.info("AutoCache.MSG_TYPE_WRITING_HOST_MEMORY_FULL")
                    chunk_idx_starting_disk_cache = msg_params['chunk_idx']
                    host_memory_window_len = chunk_idx_starting_disk_cache

            elif msg_id == Message.MSG_TYPE_WRITING_DISK_MEMORY_FULL:
                if chunk_idx_starting_recompute == -1:
                    chunk_idx_starting_recompute = msg_params['chunk_idx']
                    logging.info(
                        "AutoCache.MSG_TYPE_WRITING_DISK_MEMORY_FULL. chunk_idx_starting_recompute = %d" % chunk_idx_starting_recompute)
                if chunk_idx_starting_disk_cache == -1:
                    chunk_idx_starting_disk_cache = msg_params['chunk_idx']
                    host_memory_window_len = chunk_idx_starting_disk_cache
            elif msg_id == Message.MSG_TYPE_WRITING_END_CHUNK_IDX:
                if chunk_idx_end_cache == -1:
                    chunk_idx_end_cache = msg_params['chunk_idx']
            elif msg_id == Message.MSG_TYPE_READING:
                logging.info("AutoCache.MSG_TYPE_READING")
                # 1. find the chunk index list that should be loaded into the host memory,
                # and the chunk list that should be put into the disk storage
                chunk_idx = msg_params['chunk_idx']
                chunk_index_list_to_in_disk, chunk_index_list_to_in_memory = self.find_the_chunks_for_load_and_cache(
                    host_memory_window_len,
                    chunk_idx,
                    chunk_idx_starting_disk_cache,
                    chunk_idx_starting_recompute,
                    chunk_idx_end_cache,
                    chunk_num)
                logging.info(
                    "chunk_idx = %d, chunk_index_list_to_in_memory = %s" % (chunk_idx, str(chunk_index_list_to_in_memory)))
                logging.info(
                    "chunk_idx = %d, chunk_index_list_to_in_disk = %s" % (chunk_idx, str(chunk_index_list_to_in_disk)))

                # 2. load from disk storage
                self.load_from_pickle_file(cache_path, data_dict, chunk_index_list_to_in_memory)

                # 3. cache to disk with pickle file
                self.save_as_pickle_file(cache_path, chunk_size, chunk_idx, data_dict, chunk_index_list_to_in_disk)
            elif msg_id == Message.MSG_TYPE_TERMINATE:
                break
            elif msg_id == Message.MSG_TYPE_RESET:
                host_memory_window_len = -1
                chunk_idx_starting_disk_cache = -1
                chunk_idx_starting_recompute = -1
                chunk_idx_end_cache = -1
                chunk_num = -1
            else:
                raise Exception("no such message")
            logging.info("subprocess is running")


    def save_as_pickle_file(self, cache_path, chunk_size, chunk_idx, data_dict, chunk_index_list_to_be_cached):
        logging.info("chunk_idx = %d, chunk_index_list_to_be_cached = %s" % (chunk_idx, str(chunk_index_list_to_be_cached)))
        for chunk_idx in chunk_index_list_to_be_cached:
            loading_path = cache_path + str(chunk_idx) + ".fea"
            if not path.exists(loading_path):
                for idx in range(len(data_dict)):
                    logging.info("len of data_dict[%d] = %d" % (idx, len(data_dict[idx].keys())))
                if len(data_dict[chunk_idx]) == chunk_size:
                    numpy_queue = self.move_tensor_dict_to_numpy_dict(data_dict[chunk_idx])
                    pickle.dump(numpy_queue, open(cache_path + str(chunk_idx) + ".fea", "wb"))
                    data_dict[chunk_idx].clear()
                    logging.info("----------chunk_idx = %d save successfully----------" % chunk_idx)


    def load_from_pickle_file(self, cache_path, data_dict, chunk_index_list_to_be_loaded):
        for chunk_idx in chunk_index_list_to_be_loaded:
            loading_path = cache_path + str(chunk_idx) + ".fea"
            if path.exists(loading_path):
                logging.info("loading_path %s exists" % loading_path)
                numpy_dict = pickle.load(open(loading_path, "rb"))
                logging.info("--------chunk_idx = %d, data_dict.size() = %d" % (chunk_idx, len(data_dict)))
                self.move_numpy_dict_to_tensor_dict(data_dict, chunk_idx, numpy_dict)
                os.remove(loading_path)
                logging.info("----------chunk_idx = %d load successfully----------" % (chunk_idx))
            else:
                logging.info("loading_path %s does not exists" % loading_path)


    def move_tensor_dict_to_numpy_dict(self, tensor_dict):
        numpy_dict = dict()
        for idx in range(len(tensor_dict)):
            tensor = tensor_dict[idx]
            numpy_dict[idx] = tensor.numpy()
        return numpy_dict


    def move_numpy_dict_to_tensor_dict(self, data_dict, chunk_idx, numpy_dict):
        for idx in range(len(numpy_dict.keys())):
            numpy_feature = numpy_dict[idx]
            # print("numpy_feature = " + str(numpy_feature))
            tensor = torch.from_numpy(numpy_feature)
            data_dict[chunk_idx][idx] = tensor


    def clear_all_cache_files(self, cache_path, chunk_num):
        for chunk_idx in range(chunk_num):
            loading_path = cache_path + str(chunk_idx) + ".fea"
            if path.exists(loading_path):
                os.remove(loading_path)


    def find_the_chunks_for_load_and_cache(self, window_len, chunk_idx, chunk_idx_starting_disk_cache,
                                           chunk_idx_starting_recompute, chunk_idx_end_cache, chunk_num):
        """
            A sliding window algorithm to find the load and cache.
            THe high principle is always to keep window_len chunks in host memory.
        """
        logging.info(
            "0 - chunk_idx = %d, window_len = %d, chunk_idx_starting_disk_cache = %d, chunk_idx_starting_recompute = %d, chunk_idx_end_cache = %d, chunk_num = %d" % (
                chunk_idx, window_len, chunk_idx_starting_disk_cache,
                chunk_idx_starting_recompute, chunk_idx_end_cache, chunk_num))

        # case 1: host memory can hold all cache

        if chunk_idx_starting_disk_cache == -1 and chunk_idx_starting_recompute == -1:
            return [], []
        elif chunk_idx_starting_disk_cache != -1 and chunk_idx_starting_recompute == -1:
            return self.find_chunks_for_host_mem_full_status(window_len, chunk_idx, chunk_idx_starting_disk_cache,
                                                        chunk_idx_starting_recompute, chunk_idx_end_cache, chunk_num)

        elif chunk_idx_starting_disk_cache != -1 and chunk_idx_starting_recompute != -1:
            return self.find_chunks_for_disk_full_status(window_len, chunk_idx, chunk_idx_starting_disk_cache,
                                                    chunk_idx_starting_recompute, chunk_idx_end_cache, chunk_num)
        else:
            return [], []


    def find_chunks_for_host_mem_full_status(self, window_len, chunk_idx, chunk_idx_starting_disk_cache,
                                             chunk_idx_starting_recompute, chunk_idx_end_cache, chunk_num):
        chunk_num_in_cache = chunk_num
        chunk_index_list_to_in_disk = []
        chunk_index_list_to_in_memory = []
        if chunk_idx + window_len - 1 <= chunk_num_in_cache - 1:
            logging.info("1 - chunk_idx = %d, window_len = %d, chunk_idx_starting_recompute = %d, chunk_num = %d" % (
                chunk_idx, window_len,
                chunk_idx_starting_recompute, chunk_num))
            for idx in range(chunk_idx, chunk_idx + window_len):
                chunk_index_list_to_in_memory.append(idx)
            for idx in range(chunk_idx):
                chunk_index_list_to_in_disk.append(idx)
            for idx in range(chunk_idx + window_len, chunk_num_in_cache):
                chunk_index_list_to_in_disk.append(idx)

        elif chunk_idx + window_len - 1 > chunk_num_in_cache - 1:
            logging.info("2 - chunk_idx = %d, window_len = %d, chunk_idx_starting_recompute = %d, chunk_num = %d" % (
                chunk_idx, window_len,
                chunk_idx_starting_recompute, chunk_num))
            disk_start_chunk = chunk_idx + window_len - 1 - (chunk_num_in_cache - 1)
            disk_end_chunk = chunk_idx
            for idx in range(disk_start_chunk):
                chunk_index_list_to_in_memory.append(idx)
            for idx in range(chunk_idx, chunk_num_in_cache):
                chunk_index_list_to_in_memory.append(idx)
            for idx in range(disk_start_chunk, disk_end_chunk):
                chunk_index_list_to_in_disk.append(idx)
        return chunk_index_list_to_in_disk, chunk_index_list_to_in_memory


    def find_chunks_for_disk_full_status(self, window_len, chunk_idx, chunk_idx_starting_disk_cache,
                                         chunk_idx_starting_recompute, chunk_idx_end_cache, chunk_num):
        chunk_num_in_cache = chunk_idx_starting_recompute
        chunk_index_list_to_in_disk = []
        chunk_index_list_to_in_memory = []
        if chunk_idx + window_len - 1 <= chunk_num_in_cache - 1:
            logging.info("1 - chunk_idx = %d, window_len = %d, chunk_idx_starting_recompute = %d, chunk_num = %d" % (
                chunk_idx, window_len,
                chunk_idx_starting_recompute, chunk_num))
            for idx in range(chunk_idx, chunk_idx + window_len):
                chunk_index_list_to_in_memory.append(idx)
            for idx in range(chunk_idx):
                chunk_index_list_to_in_disk.append(idx)
            for idx in range(chunk_idx + window_len, chunk_num_in_cache):
                chunk_index_list_to_in_disk.append(idx)

        elif chunk_idx + window_len - 1 > chunk_num_in_cache - 1:
            logging.info("2 - chunk_idx = %d, window_len = %d, chunk_idx_starting_recompute = %d, chunk_num = %d" % (
                chunk_idx, window_len,
                chunk_idx_starting_recompute, chunk_num))
            disk_start_chunk = chunk_idx + window_len - 1 - (chunk_num_in_cache - 1)
            disk_end_chunk = chunk_idx
            for idx in range(disk_start_chunk):
                chunk_index_list_to_in_memory.append(idx)
            for idx in range(chunk_idx, chunk_num_in_cache):
                chunk_index_list_to_in_memory.append(idx)
            for idx in range(disk_start_chunk, disk_end_chunk):
                chunk_index_list_to_in_disk.append(idx)
        return chunk_index_list_to_in_disk, chunk_index_list_to_in_memory
