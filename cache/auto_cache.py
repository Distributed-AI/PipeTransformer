import logging
from multiprocessing import Process, Manager
from time import sleep

import torch


class AutoCache:
    def __init__(self, auto_dp, auto_pipe):
        self.auto_dp = auto_dp
        self.auto_pipe = auto_pipe
        self.num_frozen_layers = 0

        self.batch_size_train = 0
        self.chunk_num = 10

        manager = Manager()
        self.train_extracted_features = manager.dict()
        self.test_extracted_features = manager.dict()

        self.is_enable = False

        # disk storage
        self.disk_storage_process = Process(target=self.disk_process_run,
                                            args=(self.train_extracted_features,
                                                  self.test_extracted_features))
        self.disk_storage_process.start()

    def disk_process_run(self, train_extracted_features, test_extracted_features):
        while True:
            logging.info("disk_process_run")
            logging.info("train_extracted_features len = %d" % len(train_extracted_features.keys()))
            logging.info("test_extracted_features len = %d" % len(test_extracted_features.keys()))
            sleep(1)

    def update_num_frozen_layers(self, num_frozen_layers, batch_size_train, batch_size_test):
        self.num_frozen_layers = num_frozen_layers
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

        if len(self.train_extracted_features.keys()) > 0:
            for key in self.train_extracted_features.keys():
                dict = self.train_extracted_features[key]
                for key in dict.keys():
                    del dict[key]
                dict.clear()
            self.train_extracted_features.clear()

        if len(self.test_extracted_features.keys()) > 0:
            for key in self.test_extracted_features.keys():
                dict = self.test_extracted_features[key]
                for key in dict.keys():
                    del dict[key]
                dict.clear()
            self.test_extracted_features.clear()

        manager = Manager()
        if batch_size_train < self.chunk_num:
            train_extracted_dict = manager.dict()
            self.train_extracted_features[0] = train_extracted_dict
        else:
            for chunk_idx in range(self.chunk_num):
                dict_chunk_i = manager.dict()
                self.train_extracted_features[chunk_idx] = dict_chunk_i

        if batch_size_test < self.chunk_num:
            test_extracted_dict = manager.dict()
            self.test_extracted_features[0] = test_extracted_dict
        else:
            for chunk_idx in range(self.chunk_num):
                dict_chunk_i = manager.dict()
                self.test_extracted_features[chunk_idx] = dict_chunk_i

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
        chunk_idx = int(batch_idx / self.chunk_num)
        chunk_batch_idx = batch_idx % self.chunk_num
        self.train_extracted_features[chunk_idx][chunk_batch_idx] = extracted_feature.cpu()
        # self.train_extracted_features[batch_idx] = extracted_feature.cpu()

    def cache_test_extracted_hidden_feature(self, batch_idx, extracted_feature):
        if not self.is_enable:
            return
        chunk_idx = int(batch_idx / self.chunk_num)
        chunk_batch_idx = batch_idx % self.chunk_num
        self.test_extracted_features[chunk_idx][chunk_batch_idx] = extracted_feature.cpu()
        # self.train_extracted_features[batch_idx] = extracted_feature.cpu()

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

        chunk_idx = int(batch_idx / self.chunk_num)
        chunk_batch_idx = batch_idx % self.chunk_num
        return self.train_extracted_features[chunk_idx][chunk_batch_idx].to(device_idx_start)

    def get_test_extracted_hidden_feature(self, batch_idx):
        if not self.is_enable:
            return None
        if batch_idx not in self.test_extracted_features.keys():
            return None
        # the hidden features are always in device 0
        logging.info("--------get_test_extracted_hidden_feature------------")
        device_idx_start = self.auto_dp.get_local_rank() * self.auto_pipe.get_pipe_len()
        chunk_idx = int(batch_idx / self.chunk_num)
        chunk_batch_idx = batch_idx % self.chunk_num
        return self.test_extracted_features[chunk_idx][chunk_batch_idx].to(device_idx_start)
