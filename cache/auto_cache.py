import logging

import torch

from cache.auto_cache_impl import AutoCacheImpl


class AutoCache:
    def __init__(self, args, auto_dp, auto_pipe, data_manager, hidden_feature_size):
        self.auto_dp = auto_dp
        self.auto_pipe = auto_pipe
        self.data_manager = data_manager

        self.num_frozen_layers = 0
        self.batch_num_train = 0
        self.batch_num_test = 0
        self.hidden_feature_size = hidden_feature_size

        self.cache_manager_train = AutoCacheImpl(args, self.data_manager)
        self.cache_manager_test = AutoCacheImpl(args, self.data_manager)

        self.is_enable = False

    def update_sample_index(self, epoch):
        self.cache_manager_train.reset_status(epoch)
        self.cache_manager_test.reset_status(epoch)

    def update_num_frozen_layer(self, num_frozen_layers):
        self.num_frozen_layers = num_frozen_layers

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def infer_train(self, frozen_model, pipe_model, epoch, batch_idx, batch_sample_idx, x):
        if self.is_enable:
            if frozen_model is not None:
                logging.debug("infer_train. batch_idx = %d" % batch_idx)
                with torch.no_grad():
                    device_idx_start = self.auto_dp.get_local_rank() * self.auto_pipe.get_pipe_len()
                    hidden_feature = self.cache_manager_train.get_hidden_feature(
                        self.num_frozen_layers, frozen_model,
                        epoch, batch_idx, batch_sample_idx, x, device_idx_start
                    ).to(device_idx_start)
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

    def infer_test(self, frozen_model, pipe_model, epoch, batch_idx, batch_sample_idx, x):
        if self.is_enable:
            if frozen_model is not None:
                with torch.no_grad():
                    device_idx_start = self.auto_dp.get_local_rank() * self.auto_pipe.get_pipe_len()
                    hidden_feature = self.cache_manager_test.get_hidden_feature(
                        self.num_frozen_layers, frozen_model,
                        epoch, batch_idx, batch_sample_idx, x, device_idx_start
                    ).to(device_idx_start)
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
