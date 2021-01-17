import logging
import shutil

import numpy
import psutil
import torch
import torch.multiprocessing as mp

from .cache_daemon_process import CacheDaemon
from .cache_msg import Message

"""
Traceback (most recent call last):
  File "main.py", line 164, in <module>
    trainer.train_and_eval(freeze_point)
  File "/home/chaoyanghe/PipeTransformer/trainer.py", line 39, in train_and_eval
    self.train(epoch)
  File "/home/chaoyanghe/PipeTransformer/trainer.py", line 108, in train
    log_probs = self.auto_cache.infer_train(self.frozen_model, self.pipe_model, x, batch_idx)
  File "/home/chaoyanghe/PipeTransformer/cache/auto_cache.py", line 44, in infer_train
    hidden_feature = self.two_level_cache_train.get_hidden_feature(batch_idx, x, frozen_model).to(
  File "/home/chaoyanghe/PipeTransformer/cache/two_level_cache.py", line 232, in get_hidden_feature
    hidden_feature = self.write_one_batch(batch_idx, x, model)
  File "/home/chaoyanghe/PipeTransformer/cache/two_level_cache.py", line 251, in write_one_batch
    self.data_dict[chunk_idx][chunk_batch_idx] = hidden_feature
  File "<string>", line 2, in __setitem__
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/managers.py", line 834, in _callmethod
    raise convert_to_error(kind, result)
multiprocessing.managers.RemoteError: 
---------------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/managers.py", line 234, in serve_client
    request = recv()
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/connection.py", line 251, in recv
    return _ForkingPickler.loads(buf.getbuffer())
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 282, in rebuild_storage_fd
    fd = df.detach()
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/resource_sharer.py", line 58, in detach
    return reduction.recv_handle(conn)
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/reduction.py", line 185, in recv_handle
    return recvfds(s, 1)[0]
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/reduction.py", line 161, in recvfds
    len(ancdata))
RuntimeError: received 0 items of ancdata


When disk is OOM:

Traceback (most recent call last):
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/queues.py", line 236, in _feed
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 321, in reduce_storage
RuntimeError: unable to open shared memory object </torch_52868_1950128645> in read-write mode

When a chunk is too big to hold by a pickle file:

Traceback (most recent call last):
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/home/chaoyanghe/anaconda3/envs/pipe_ditributed/lib/python3.7/multiprocessing/process.py", line 99, in run
    self._target(*self._args, **self._kwargs)
  File "test.py", line 28, in disk_cache_process_impl
    save_as_pickle_file(path, chunk_idx, data_queue_list, chunk_index_list_to_be_cached)
  File "test.py", line 91, in save_as_pickle_file
    pickle.dump(numpy_queue, open(path + str(chunk_idx) + ".fea", "wb"))
OverflowError: cannot serialize a bytes object larger than 4 GiB
"""


class AutoCacheImplWithHostMem:

    def __init__(self, args, data_manager):
        self.args = args
        self.data_manager = data_manager
        self.epoch = 0

        self.host_memory_percentage = 0.2
        self.disk_memory_percentage = 0.2

        self.msg_q = mp.Queue()

        self.cache_daemon = CacheDaemon(args, self.msg_q)
        self.cache_daemon.daemon = True
        self.cache_daemon.start()

        self.sample_uid_to_feature_dict = dict()
        self.sample_uid_to_layer_id_dict = dict()
        self.dtype = None

    def reset_status(self, epoch):
        train_sample_index = self.data_manager.get_train_sample_index(epoch)
        test_sample_index = self.data_manager.get_test_sample_index(epoch)
        msg = Message(Message.MSG_TYPE_UPDATE_INDEX)
        msg.set(Message.MSG_KEY_EPOCH, epoch)
        msg.set(Message.MSG_KEY_TRAIN_SAMPLE_INDEX, train_sample_index)
        msg.set(Message.MSG_KEY_TEST_SAMPLE_INDEX, test_sample_index)
        self.msg_q.put(msg)

    def get_hidden_feature(self, num_frozen_layer, model, epoch, batch_idx, batch_sample_idx, x, device):
        logging.info("(global_rank = %d) get_hidden_feature. epoch = %d, batch_idx = %d, batch_sample_idx = %s" % (
        self.args.global_rank, epoch, batch_idx, str("")))

        b_is_batch_cached = True
        layer_id = 0
        if self._is_batch_in_cache(batch_sample_idx):
            logging.info("(global_rank = %d) copy from shared memory START" % self.args.global_rank)

            sample_idx_in_batch = 0
            hidden_tensor_np = numpy.ndarray(
                [self.args.batch_size, self.args.seq_len, self.args.transformer_hidden_size],
                dtype=self.dtype
            )
            for sample_uid in batch_sample_idx:
                cache_hidden_feature = self.sample_uid_to_feature_dict[sample_uid]
                layer_id = self.sample_uid_to_layer_id_dict[sample_uid]
                hidden_tensor_np[sample_idx_in_batch] = cache_hidden_feature[:]
                sample_idx_in_batch += 1
            if hidden_tensor_np is not None:
                hidden_feature = torch.from_numpy(hidden_tensor_np).cpu()
        else:
            b_is_batch_cached = False

        if b_is_batch_cached:
            logging.info("(global_rank = %d) get_hidden_feature. NO need to compute FP (layer 0-%d), "
                         "only compute FP (layer %d-%d), get from shared memory" % (
                             self.args.global_rank, layer_id - 1, layer_id, num_frozen_layer))
            logging.info("(global_rank = %d) copy from shared memory END" % self.args.global_rank)
            if layer_id != num_frozen_layer:
                with torch.no_grad():
                    hidden_feature_without_cache = model(x).detach().cpu()
                    hidden_feature = model(hidden_feature.to(device), layer_id).detach().cpu()
                    if not torch.equal(hidden_feature_without_cache, hidden_feature):
                        raise Exception("not equal with inference from layer 0")
                    else:
                        logging.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                self._cache_a_batch_sample(batch_sample_idx, hidden_feature, num_frozen_layer)
                logging.info("(global_rank = %d) update shared memory END" % self.args.global_rank)
        else:
            logging.info(
                "(global_rank = %d) get_hidden_feature. cache to shared memory (START)" % self.args.global_rank)
            # [60, 197, 768]
            with torch.no_grad():
                hidden_feature = model(x).detach().cpu()
                if self.dtype is None:
                    self.dtype = hidden_feature.numpy().dtype
            self._cache_a_batch_sample(batch_sample_idx, hidden_feature, num_frozen_layer)
            logging.info("(global_rank = %d) get_hidden_feature. cache to shared memory (END)" % self.args.global_rank)

        self._send_training_progress_to_daemon(epoch, batch_idx)
        return hidden_feature

    def _is_batch_in_cache(self, batch_sample_idx):
        for sample_uid in batch_sample_idx:
            if sample_uid not in self.sample_uid_to_feature_dict.keys():
                return False
        return True

    def _cache_a_batch_sample(self, batch_sample_idx, hidden_feature, num_frozen_layer):
        sample_idx_in_batch = 0
        for sample_uid in batch_sample_idx:
            # [197, 768]
            sample = hidden_feature[sample_idx_in_batch, :, :]
            self.sample_uid_to_feature_dict[sample_uid] = sample
            self.sample_uid_to_layer_id_dict[sample_uid] = num_frozen_layer
            sample_idx_in_batch += 1

    def _send_training_progress_to_daemon(self, epoch, batch_idx):
        logging.info("_send_training_progress_to_daemon. epoch = %d, batch_idx = %d" % (epoch, batch_idx))
        msg = Message(Message.MSG_TYPE_TRAINING_PROGRESS)
        msg.set(Message.MSG_KEY_EPOCH, epoch)
        msg.set(Message.MSG_KEY_BATCH_INDEX, batch_idx)
        self.msg_q.put(msg)

    def is_disk_storage_full(self):
        total, used, free = shutil.disk_usage(__file__)
        used_storage_percentage = used / total
        # logging.info("is_disk_storage_full. Percentage = " + str(used_storage_percentage))
        return True if used_storage_percentage > self.disk_memory_percentage else False

    def is_host_memory_full(self):
        memory_cost_percent = 1 - psutil.virtual_memory()[4] / psutil.virtual_memory()[0]
        # logging.info("is_host_memory_full. Percentage = " + str(memory_cost_percent))
        return True if memory_cost_percent > self.host_memory_percentage else False
