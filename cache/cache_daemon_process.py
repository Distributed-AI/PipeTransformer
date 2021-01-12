import logging

import torch.multiprocessing as mp

from cache.cache_msg import Message
from cache.disk_memory_manager import DiskMemoryManager
from cache.shared_memory_manager import SharedMemoryManager
from cache.shared_memory_manager_int_value import SharedMemoryManagerIntValue


class CacheDaemon(mp.Process):
    def __init__(self, args, msg_q):
        super().__init__()
        self.msg_q = msg_q
        self.shared_memory_mgr_hidden_feature = SharedMemoryManager(args, "hidden_feature")

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
                batch_sample_idx = message.get(Message.MSG_KEY_BATCH_SAMPLE_INDEX)
                hidden_feature = message.get(Message.MSG_KEY_HIDDEN_FEATURE)
                num_frozen_layer = message.get(Message.MSG_KEY_NUM_FROZEN_LAYER)
                cached_layer_id = message.get(Message.MSG_KEY_CACHED_NUM_FROZEN_LAYER)

                # add new tensor to cache, and delete the old ones
                if cached_layer_id > 0:
                    self._delete_previous_cached_batch(batch_sample_idx, cached_layer_id)
                self._cache_a_batch_sample(batch_sample_idx, hidden_feature, num_frozen_layer)

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

    def _cache_a_batch_sample(self, batch_sample_idx, hidden_feature, num_frozen_layer):
        sample_idx_in_batch = 0
        for sample_uid in batch_sample_idx:
            # [197, 768]
            sample = hidden_feature[sample_idx_in_batch, :, :]
            self.shared_memory_mgr_hidden_feature.add_tensor(sample_uid, num_frozen_layer, sample)
            sample_idx_in_batch += 1

    def _delete_previous_cached_batch(self, batch_sample_idx, cached_layer_id):
        sample_idx_in_batch = 0
        for sample_uid in batch_sample_idx:
            self.shared_memory_mgr_hidden_feature.delete_tensor(sample_uid, cached_layer_id)
            sample_idx_in_batch += 1

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
