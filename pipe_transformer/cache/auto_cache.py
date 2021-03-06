import logging

import torch

from .auto_cache_impl import AutoCacheImpl


class AutoCache:
    def __init__(self, config, auto_freeze, auto_dp, auto_pipe, data_manager):
        self.auto_freeze = auto_freeze
        self.auto_dp = auto_dp
        self.auto_pipe = auto_pipe
        self.data_manager = data_manager

        self.num_frozen_layers = 0

        # self.cache_manager_train = AutoCacheImplWithHostMem(args, self.data_manager)
        # self.cache_manager_test = AutoCacheImplWithHostMem(args, self.data_manager)

        self.cache_manager = AutoCacheImpl(config, self.data_manager)

        self.is_enable = config.b_cache

    def update_sample_index(self, epoch):
        self.cache_manager.reset_status(epoch)

    def update_num_frozen_layer(self, num_frozen_layers):
        self.num_frozen_layers = num_frozen_layers

    def forward_with_cache(self, frozen_model, pipe_model, epoch, batch_idx, batch_sample_idx, x, is_train_mode, is_train_data):
        if self.num_frozen_layers != self.auto_pipe.get_num_frozen_layers():
            raise Exception("num_frozen_layers does not match with the pipe")
        if self.is_enable and self.num_frozen_layers >= 3:
            if frozen_model is not None:
                logging.debug("infer_train. batch_idx = %d" % batch_idx)
                with torch.no_grad():
                    device_idx_start = self.auto_dp.get_local_rank() * self.auto_pipe.get_pipe_len()
                    hidden_feature = self.cache_manager.get_hidden_feature(
                        self.auto_freeze.get_num_of_frozen_layer(epoch - 1 if epoch - 1 >= 0 else 0),
                        self.num_frozen_layers, frozen_model,
                        epoch, batch_idx, batch_sample_idx, x, device_idx_start, is_train_mode, is_train_data
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

    def cleanup(self):
        self.cache_manager.cleanup()
