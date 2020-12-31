import logging
from multiprocessing import Process, Queue

import torch

from cache.disk_storage_process import DiskStorageProcess


class AutoCache:
    def __init__(self, auto_dp, auto_pipe):
        self.auto_dp = auto_dp
        self.auto_pipe = auto_pipe
        self.num_frozen_layers = 0
        self.train_extracted_features = dict()
        self.test_extracted_features = dict()

        self.is_enable = False

        # # disk storage
        # pqueue = Queue()
        # self.disk_storage_thread = DiskStorageProcess(pqueue, self.train_extracted_features,
        #                                              self.test_extracted_features)
        # self.disk_storage_thread.start()

    def update_num_frozen_layers(self, num_frozen_layers):
        self.num_frozen_layers = num_frozen_layers
        self.train_extracted_features.clear()
        self.test_extracted_features.clear()

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def infer_train(self, frozen_model, pipe_model, x, batch_idx):
        if self.is_enable:
            if frozen_model is not None:
                if self.get_train_extracted_hidden_feature(batch_idx) is None:
                    with torch.no_grad():
                        hidden_feature = frozen_model(x)
                    self.cache_train_extracted_hidden_feature(batch_idx, hidden_feature)
                else:
                    hidden_feature = self.get_train_extracted_hidden_feature(batch_idx)
                log_probs = pipe_model(hidden_feature)
            else:
                log_probs = pipe_model(x)
        else:
            if frozen_model is None:
                log_probs = pipe_model(x)
            else:
                with torch.no_grad():
                    hidden_feature = frozen_model(x)
                log_probs = pipe_model(hidden_feature)
        return log_probs

    def infer_test(self, frozen_model, pipe_model, x, batch_idx):
        if self.is_enable:
            if frozen_model is not None:
                if self.get_test_extracted_hidden_feature(batch_idx) is None:
                    with torch.no_grad():
                        hidden_feature = frozen_model(x)
                    self.cache_test_extracted_hidden_feature(batch_idx, hidden_feature)
                else:
                    hidden_feature = self.get_test_extracted_hidden_feature(batch_idx)
                log_probs = pipe_model(hidden_feature)
            else:
                log_probs = pipe_model(x)
        else:
            if frozen_model is None:
                log_probs = pipe_model(x)
            else:
                with torch.no_grad():
                    hidden_feature = frozen_model(x)
                log_probs = pipe_model(hidden_feature)
        return log_probs

    #
    # def cache_train_extracted_hidden_feature(self, batch_idx, extracted_feature):
    #     if not self.is_enable:
    #         return
    #     self.train_extracted_features[batch_idx] = extracted_feature
    #
    # def cache_test_extracted_hidden_feature(self, batch_idx, extracted_feature):
    #     if not self.is_enable:
    #         return
    #     self.test_extracted_features[batch_idx] = extracted_feature

    def cache_train_extracted_hidden_feature(self, batch_idx, extracted_feature):
        if not self.is_enable:
            return
        self.train_extracted_features[batch_idx] = extracted_feature.cpu()

    def cache_test_extracted_hidden_feature(self, batch_idx, extracted_feature):
        if not self.is_enable:
            return
        self.test_extracted_features[batch_idx] = extracted_feature.cpu()

    # def get_train_extracted_hidden_feature(self, batch_idx):
    #     if not self.is_enable:
    #         return None
    #     # the hidden features are always in device 0
    #     if batch_idx not in self.train_extracted_features.keys():
    #         return None
    #     logging.info("--------get_train_extracted_hidden_feature------------")
    #     return self.train_extracted_features[batch_idx]
    #
    # def get_test_extracted_hidden_feature(self, batch_idx):
    #     if not self.is_enable:
    #         return None
    #     if batch_idx not in self.test_extracted_features.keys():
    #         return None
    #     # the hidden features are always in device 0
    #     logging.info("--------get_test_extracted_hidden_feature------------")
    #     return self.test_extracted_features[batch_idx]

    def get_train_extracted_hidden_feature(self, batch_idx):
        if not self.is_enable:
            return None
        # the hidden features are always in device 0
        if batch_idx not in self.train_extracted_features.keys():
            return None
        logging.info("--------get_train_extracted_hidden_feature------------")
        device_idx_start = self.auto_dp.get_local_rank() * self.auto_pipe.get_pipe_len()
        return self.train_extracted_features[batch_idx].to(device_idx_start)

    def get_test_extracted_hidden_feature(self, batch_idx):
        if not self.is_enable:
            return None
        if batch_idx not in self.test_extracted_features.keys():
            return None
        # the hidden features are always in device 0
        logging.info("--------get_test_extracted_hidden_feature------------")
        device_idx_start = self.auto_dp.get_local_rank() * self.auto_pipe.get_pipe_len()
        return self.test_extracted_features[batch_idx].to(device_idx_start)
