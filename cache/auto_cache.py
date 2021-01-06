import logging

import torch

from cache.one_level_cache import OneLevelCache


class AutoCache:
    def __init__(self, auto_dp, auto_pipe, hidden_feature_size):
        self.auto_dp = auto_dp
        self.auto_pipe = auto_pipe

        self.num_frozen_layers = 0
        self.batch_num_train = 0
        self.batch_num_test = 0
        self.hidden_feature_size = hidden_feature_size

        self.two_level_cache_train = OneLevelCache()
        self.two_level_cache_test = OneLevelCache()

        self.is_enable = False

    def reset(self, num_frozen_layers, batch_num_train, batch_num_test):
        self.num_frozen_layers = num_frozen_layers
        self.batch_num_train = batch_num_train
        self.batch_num_test = batch_num_test
        self.two_level_cache_train.reset_status(False, self.batch_num_train,
                                                self.hidden_feature_size, self.auto_dp.get_active_world_size())
        self.two_level_cache_test.reset_status(False, self.batch_num_test,
                                               self.hidden_feature_size, self.auto_dp.get_active_world_size())

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def infer_train(self, frozen_model, pipe_model, x, batch_idx, epoch):
        if self.is_enable:
            if frozen_model is not None:
                logging.debug("infer_train. batch_idx = %d" % batch_idx)
                with torch.no_grad():
                    device_idx_start = self.auto_dp.get_local_rank() * self.auto_pipe.get_pipe_len()
                    hidden_feature = self.two_level_cache_train.get_hidden_feature(epoch, batch_idx, x, frozen_model).to(
                        device_idx_start)
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

    def infer_test(self, frozen_model, pipe_model, x, batch_idx, epoch):
        if self.is_enable:
            if frozen_model is not None:
                with torch.no_grad():
                    device_idx_start = self.auto_dp.get_local_rank() * self.auto_pipe.get_pipe_len()
                    hidden_feature = self.two_level_cache_test.get_hidden_feature(epoch, batch_idx, x, frozen_model).to(
                        device_idx_start)
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
