import logging
import math
import shutil

import psutil
import torch
import wandb

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


class OneLevelCache:
    MSG_TYPE_WRITING = 0
    MSG_TYPE_WRITING_END_CHUNK_IDX = 1
    MSG_TYPE_WRITING_HOST_MEMORY_FULL = 2
    MSG_TYPE_WRITING_DISK_MEMORY_FULL = 3
    MSG_TYPE_READING = 4
    MSG_TYPE_TERMINATE = 5
    MSG_TYPE_RESET = 6

    def __init__(self):
        self.cache_path = "./hidden_feature_cache_" + str(id(self))
        self.is_enable = False

        self.batch_size = -1

        self.host_memory_percentage = 0.8

        self.data_dict = dict()

        self.is_cache_ready = False

    def reset_status(self, is_ready, batch_size, hidden_feature_size, processes_num):
        self.is_cache_ready = is_ready
        self.data_dict.clear()
        self.batch_size = batch_size
        logging.info("self.is_cache_ready = %s, self.batch_size = %s" % (str(self.is_cache_ready), str(self.batch_size)))

    def get_hidden_feature(self, epoch, batch_idx, x, model):
        if not self.is_cache_ready:
            hidden_feature = self.write_one_batch(epoch, batch_idx, x, model)
        else:
            hidden_feature = self.read_one_batch(epoch, batch_idx, x, model)
        return hidden_feature

    def write_one_batch(self, epoch, batch_idx, x, model):
        hidden_feature = model(x).detach().cpu()

        if not self.is_host_memory_full():
            logging.info("####################host memory is not full. epoch = %d, batch_idx = %d" % (epoch, batch_idx))
            self.data_dict[batch_idx] = hidden_feature
        if batch_idx == self.batch_size-1:
            logging.info("####################write_one_batch finished. epoch = %d, batch_idx = %d" % (epoch, batch_idx))
            self.is_cache_ready = True
        return hidden_feature

    def read_one_batch(self, epoch, batch_idx, x, model):
        if batch_idx in self.data_dict.keys():
            logging.info("####################read from cache. epoch = %d, batch_idx = %d" % (epoch, batch_idx))
            hidden_feature = self.data_dict[batch_idx]
        else:
            hidden_feature = model(x)
        return hidden_feature

    def is_host_memory_full(self):
        memory_cost_percent = 1 - psutil.virtual_memory()[4] / psutil.virtual_memory()[0]
        # logging.info("is_host_memory_full. Percentage = " + str(memory_cost_percent))
        return True if memory_cost_percent > self.host_memory_percentage else False