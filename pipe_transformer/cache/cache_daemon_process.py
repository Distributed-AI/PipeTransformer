import logging
import shutil

import psutil
import torch.multiprocessing as mp

from .cache_msg import Message
from .disk_memory_manager import DiskMemoryManager
from .shared_memory_manager import SharedMemoryManager


class CacheDaemon(mp.Process):
    def __init__(self, config, msg_q):
        super().__init__()
        self.msg_q = msg_q
        self.shared_memory_mgr_hidden_feature_train = SharedMemoryManager(config, "hidden_feature_train")
        self.shared_memory_mgr_hidden_feature_test = SharedMemoryManager(config, "hidden_feature_test")

        self.disk_memory_mgr = DiskMemoryManager("hidden_feature")

        self.epoch = 0
        self.train_sample_index = []
        self.test_sample_index = []

        self.host_memory_percentage = 0.65
        self.disk_memory_percentage = 0.85


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
                # self._delete_previous_cached_batch(batch_sample_idx, cached_layer_id)
                self._cache_a_batch_sample(cached_layer_id, batch_sample_idx, hidden_feature, num_frozen_layer, True)

                sample_index_list_to_disk, \
                sample_index_list_to_memory = self._determine_sample_location_with_sliding_window(epoch, batch_idx)
                self._move_shared_memory_to_disk(sample_index_list_to_disk)
                self._move_disk_memory_to_shared_memory(sample_index_list_to_memory)

            elif msg_type == Message.MSG_TYPE_TEST_PROGRESS:
                logging.info("Message.MSG_TYPE_TEST_PROGRESS")
                epoch = message.get(Message.MSG_KEY_EPOCH)
                batch_idx = message.get(Message.MSG_KEY_BATCH_INDEX)
                batch_sample_idx = message.get(Message.MSG_KEY_BATCH_SAMPLE_INDEX)
                hidden_feature = message.get(Message.MSG_KEY_HIDDEN_FEATURE)
                num_frozen_layer = message.get(Message.MSG_KEY_NUM_FROZEN_LAYER)
                cached_layer_id = message.get(Message.MSG_KEY_CACHED_NUM_FROZEN_LAYER)

                # add new tensor to cache, and delete the old ones
                # self._delete_previous_cached_batch(batch_sample_idx, cached_layer_id)
                self._cache_a_batch_sample(cached_layer_id, batch_sample_idx, hidden_feature, num_frozen_layer, False)

                sample_index_list_to_disk, \
                sample_index_list_to_memory = self._determine_sample_location_with_sliding_window(epoch, batch_idx)
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
            elif msg_type == Message.MSG_TYPE_FINISH:
                self.shared_memory_mgr_hidden_feature_train.cleanup()
                self.shared_memory_mgr_hidden_feature_test.cleanup()
                break
            else:
                raise Exception("no such message")
            logging.info("subprocess is running")

    def _determine_sample_location_with_sliding_window(self, epoch, current_batch_idx):
        sample_index_list_to_disk = []
        sample_index_list_to_memory = []
        return sample_index_list_to_disk, sample_index_list_to_memory

    def _cache_a_batch_sample(self, cached_layer_id, batch_sample_idx, hidden_feature, num_frozen_layer, is_train):
        if self._is_host_memory_full():
            return
        if cached_layer_id > num_frozen_layer:
            raise Exception("cached_layer_id illegal")
        if is_train:
            shared_memory_mgr = self.shared_memory_mgr_hidden_feature_train
        else:
            shared_memory_mgr = self.shared_memory_mgr_hidden_feature_test
        sample_idx_in_batch = 0
        for sample_uid in batch_sample_idx:
            # [197, 768]
            sample = hidden_feature[sample_idx_in_batch, :, :]
            shared_memory_mgr.add_tensor(sample_uid, num_frozen_layer, sample)
            sample_idx_in_batch += 1
        logging.info("successfully!")

    def _delete_previous_cached_batch(self, batch_sample_idx, cached_layer_id):
        sample_idx_in_batch = 0
        for sample_uid in batch_sample_idx:
            self.shared_memory_mgr_hidden_feature_train.delete_tensor(sample_uid, cached_layer_id)
            sample_idx_in_batch += 1

    def _is_disk_storage_full(self):
        total, used, free = shutil.disk_usage(__file__)
        used_storage_percentage = used / total
        # logging.info("is_disk_storage_full. Percentage = " + str(used_storage_percentage))
        return True if used_storage_percentage > self.disk_memory_percentage else False

    def _is_host_memory_full(self):
        memory_cost_percent = 1 - psutil.virtual_memory()[4] / psutil.virtual_memory()[0]
        # logging.info("is_host_memory_full. Percentage = " + str(memory_cost_percent))
        return True if memory_cost_percent > self.host_memory_percentage else False

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
