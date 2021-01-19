import logging

from pipe_transformer.cache.auto_cache import AutoCache
from pipe_transformer.dp.auto_dp import AutoDataParallel
from pipe_transformer.freeze.auto_freeze import AutoFreeze
from pipe_transformer.pipe.auto_pipe import AutoElasticPipe


class PipeTransformer:
    def __init__(self, config, data_manager, model):
        self.config = config
        self.data_manager = data_manager
        self.model = model

        self.auto_dp = AutoDataParallel(config)
        config.world_size = self.auto_dp.get_world_size()
        config.global_rank = self.auto_dp.get_global_rank()

        self.auto_freeze = AutoFreeze(config)
        self.auto_pipe = AutoElasticPipe(config, model)
        self.auto_cache = AutoCache(config, self.auto_freeze, self.auto_dp, self.auto_pipe, data_manager)

        self.frozen_model, self.pipe_model = None, None
        self.train_dl, self.test_dl = None, None
        self.device_first, self.device_last = None, None

        self.epoch_start = 0

    def start(self):
        freeze_point = dict()
        freeze_point['epoch'] = 0
        frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed = self.auto_dp.transform(self.auto_pipe,
                                                                                                        self.auto_freeze,
                                                                                                        None,
                                                                                                        self.model,
                                                                                                        0, freeze_point)
        self.frozen_model = frozen_model
        self.pipe_model = pipe_model

        freeze_point = self.auto_dp.get_freeze_point()
        self.epoch_start = freeze_point['epoch']
        self._update_data_and_cache(0, True, True)
        return self.epoch_start

    def transform(self, epoch):
        if self.auto_freeze.is_freeze_open():
            new_freeze_point = dict()
            new_freeze_point['epoch'] = epoch

            frozen_layer_idx = self.auto_freeze.freeze(epoch)
            self.frozen_model, self.pipe_model, \
            is_pipe_len_changed, is_frozen_layer_changed = self.auto_dp.transform(self.auto_pipe,
                                                                                  self.auto_freeze,
                                                                                  self.frozen_model,
                                                                                  self.pipe_model,
                                                                                  frozen_layer_idx,
                                                                                  new_freeze_point)
            self._update_data_and_cache(epoch, is_pipe_len_changed, is_frozen_layer_changed)

        self.device_first = self.auto_pipe.get_device_first()
        self.device_last = self.auto_pipe.get_device_last()

    def get_new_model_and_dataset(self):
        return self.frozen_model, self.pipe_model, self.train_dl, self.test_dl, self.device_first, self.device_last

    def forward(self, epoch, batch_idx, sample_index_list, x, is_train_mode, is_train_data):
        log_probs = self.auto_cache.forward_with_cache(self.frozen_model, self.pipe_model,
                                                       epoch, batch_idx, sample_index_list, x, is_train_mode, is_train_data)
        return log_probs

    def collect_freeze_info(self):
        if self.auto_freeze.is_freeze_open():
            self.auto_freeze.accumulate(self.auto_pipe.get_origin_model())

    def _update_data_and_cache(self, epoch, is_pipe_len_changed, is_frozen_layer_changed):
        if is_pipe_len_changed:
            # epoch, batch_size, node_rank, num_replicas, local_rank
            self.train_dl, self.test_dl = self.data_manager.get_data_loader_with_node_rank(
                epoch,
                self.config.batch_size,
                self.config.node_rank,
                self.auto_dp.get_local_data_duplicate_num(),
                self.auto_dp.get_local_rank()
            )

            logging.info("global_rank = %d. is_frozen_layer_changed: %s" % (
                self.auto_dp.get_global_rank(), str(is_frozen_layer_changed)))
            """
            To help the cache to adjust the shared_memory and the disk memory, 
            we need to update the cache index when the pipe len has been changed
            """
            self.auto_cache.update_sample_index(epoch)

            # synchronize the parameters to all newly created pipes

        if is_frozen_layer_changed:
            self.auto_cache.update_num_frozen_layer(self.auto_pipe.get_num_frozen_layers())

    def get_global_rank(self):
        return self.auto_dp.get_global_rank()

    def get_active_world_size(self):
        return self.auto_dp.get_active_world_size()

    def finish(self):
        self.auto_cache.cleanup()
        self.auto_freeze.cleanup()
        self.auto_dp.cleanup()
