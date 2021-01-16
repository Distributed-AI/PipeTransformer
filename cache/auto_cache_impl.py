import argparse
import copy
import logging
import shutil
import time

import numpy
import psutil
import torch
import torch.multiprocessing as mp

from cache.cache_daemon_process import CacheDaemon
from cache.cache_msg import Message
from cache.shared_memory_manager import SharedMemoryManager
from data_preprocessing.cv_data_manager import CVDatasetManager

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


class AutoCacheImpl:

    def __init__(self, args, data_manager):
        self.args = args
        self.data_manager = data_manager

        self.msg_q = mp.Queue()

        self.cache_daemon = CacheDaemon(args, self.msg_q)
        self.cache_daemon.daemon = True
        self.cache_daemon.start()

        self.shared_memory_mgr_hidden_feature_train = SharedMemoryManager(self.args, "hidden_feature_train")
        self.shared_memory_mgr_hidden_feature_test = SharedMemoryManager(self.args, "hidden_feature_test")

        self.count_mismatch = 0

    def reset_status(self, epoch):
        train_sample_index = self.data_manager.get_train_sample_index(epoch)
        test_sample_index = self.data_manager.get_test_sample_index(epoch)
        msg = Message(Message.MSG_TYPE_UPDATE_INDEX)
        msg.set(Message.MSG_KEY_EPOCH, epoch)
        msg.set(Message.MSG_KEY_TRAIN_SAMPLE_INDEX, train_sample_index)
        msg.set(Message.MSG_KEY_TEST_SAMPLE_INDEX, test_sample_index)
        self.msg_q.put(msg)

    def cleanup(self):
        msg = Message(Message.MSG_TYPE_FINISH)
        self.msg_q.put(msg)

        self.cache_daemon.join()

        self.cache_daemon.terminate()
        self.cache_daemon.kill()
        self.msg_q.close()

    def watch_dog_process_impl(self, msg_q):
        while True:
            try:
                msg = msg_q.get(timeout=2.0)
            except msg_q.Empty as e:
                print("[WATCHDOG]: Maybe WORKER is slacking")
                msg_q.put("KILL WORKER")

    def get_hidden_feature(self, num_frozen_layer_last_epoch, num_frozen_layer, model, epoch, batch_idx,
                           batch_sample_idx, x, device, is_train_mode, is_train_data):
        if is_train_mode:
            cached_num_frozen_layer = num_frozen_layer_last_epoch
        else:
            cached_num_frozen_layer = num_frozen_layer
        hidden_feature = self._get_a_cached_batch_sample(cached_num_frozen_layer, batch_sample_idx, is_train_data)
        if hidden_feature is not None:
            logging.critical("(global_rank = %s, epoch = %s, batch_idx = %s, is_train_mode = %s, is_train_data = %s, "
                             "num_frozen_layer_last_epoch = %s, num_frozen_layer = %s) "
                             "NO need to compute FP (layer 0-%d), "
                             "frozen layer number = %d (START)"
                             % (str(self.args.global_rank), str(epoch), str(batch_idx), str(is_train_mode), str(is_train_data),
                                str(num_frozen_layer_last_epoch), str(num_frozen_layer),
                                cached_num_frozen_layer - 1, num_frozen_layer))

            if self.args.is_debug_mode:
                self._check_the_tensor_during_debug_mode(model, x, batch_idx, hidden_feature,
                                                         cached_num_frozen_layer, device)

            if num_frozen_layer > cached_num_frozen_layer:
                hidden_feature = model(hidden_feature.to(device), cached_num_frozen_layer).detach().cpu()
                self._send_to_daemon_for_cache(epoch, batch_idx, batch_sample_idx, hidden_feature,
                                               cached_num_frozen_layer, num_frozen_layer, is_train_data)
                logging.critical("(global_rank = %d) cached layer %d" % (self.args.global_rank, num_frozen_layer))
            logging.critical("(global_rank = %d) NO need to compute FP (END)" % self.args.global_rank)
        else:
            logging.critical("(global_rank = %d, epoch = %d, batch_idx = %d, is_train_mode = %s, is_train_data = %s, "
                             "num_frozen_layer_last_epoch = %d, num_frozen_layer = %d) "
                             "cache to shared memory (START)"
                             % (self.args.global_rank, epoch, batch_idx, str(is_train_mode), str(is_train_data),
                                num_frozen_layer_last_epoch, num_frozen_layer))
            with torch.no_grad():
                hidden_feature = model(x).detach().cpu()
            self._send_to_daemon_for_cache(epoch, batch_idx, batch_sample_idx, hidden_feature,
                                           cached_num_frozen_layer, num_frozen_layer, is_train_data)
            logging.critical("(global_rank = %d) cache to shared memory (END)" % self.args.global_rank)
        return hidden_feature

    def _check_the_tensor_during_debug_mode(self, model, x, batch_idx, hidden_feature, num_frozen_layer_last_epoch,
                                            device):
        # check correctness
        with torch.no_grad():
            hidden_feature_without_cache = model(x).detach().cpu()
            hidden_feature = model(hidden_feature.to(device), num_frozen_layer_last_epoch).detach().cpu()
            if not torch.equal(hidden_feature_without_cache, hidden_feature):
                logging.info(
                    "(global_rank = %d, batch_idx = %d) tensor does not match" % (self.args.global_rank, batch_idx))
                self.count_mismatch += 1
                logging.info("self.count_mismatch = %d" % self.count_mismatch)
                raise Exception("not equal with inference from layer 0")
            else:
                logging.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    def _get_a_cached_batch_sample(self, num_frozen_layer_last_epoch, batch_sample_idx, is_train):
        if is_train:
            shared_memory_mgr_hidden_feature = self.shared_memory_mgr_hidden_feature_train
        else:
            shared_memory_mgr_hidden_feature = self.shared_memory_mgr_hidden_feature_test
        sample_idx_in_batch = 0
        hidden_tensor_np = numpy.ndarray(
            (self.args.batch_size, self.args.seq_len, self.args.transformer_hidden_size),
            dtype=numpy.float32
        )
        for sample_uid in batch_sample_idx:
            cache_hidden_feature = shared_memory_mgr_hidden_feature.get_tensor(sample_uid, num_frozen_layer_last_epoch)
            if cache_hidden_feature is None:
                logging.info("tensor is none")
                return None
            hidden_tensor_np[sample_idx_in_batch] = copy.deepcopy(cache_hidden_feature)
            sample_idx_in_batch += 1
        return torch.from_numpy(hidden_tensor_np).cpu()

    def _send_to_daemon_for_cache(self, epoch, batch_idx, batch_sample_idx, hidden_feature, cached_layer_id,
                                  num_frozen_layer, is_train):
        logging.info("_send_training_progress_to_daemon. epoch = %d, batch_idx = %d" % (epoch, batch_idx))
        if is_train:
            msg = Message(Message.MSG_TYPE_TRAINING_PROGRESS)
        else:
            msg = Message(Message.MSG_TYPE_TEST_PROGRESS)
        msg.set(Message.MSG_KEY_EPOCH, epoch)
        msg.set(Message.MSG_KEY_BATCH_INDEX, batch_idx)
        msg.set(Message.MSG_KEY_BATCH_SAMPLE_INDEX, batch_sample_idx)
        msg.set(Message.MSG_KEY_HIDDEN_FEATURE, hidden_feature)
        msg.set(Message.MSG_KEY_NUM_FROZEN_LAYER, num_frozen_layer)
        msg.set(Message.MSG_KEY_CACHED_NUM_FROZEN_LAYER, cached_layer_id)
        self.msg_q.put(msg)

class Trainer:
    def __init__(self, two_level_cache, model):
        self.auto_cache = two_level_cache
        self.model = model

    def train(self):
        n_test_batches = 10
        x = torch.randn(500, 197, 768)
        for batch_idx in range(n_test_batches):
            hidden_feature = auto_cache.get_hidden_feature(0, batch_idx, x, self.model)

        logging.info("---------------------\n")
        logging.info("---------------------\n")
        logging.info("----------READ 1-----------\n")
        for batch_idx in range(n_test_batches):
            hidden_feature = auto_cache.get_hidden_feature(1, batch_idx, x, self.model)

        logging.info("---------------------\n")
        logging.info("---------------------\n")
        logging.info("----------READ 2-----------\n")

        for batch_idx in range(n_test_batches):
            hidden_feature = auto_cache.get_hidden_feature(2, batch_idx, x, self.model)


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PipeTransformer: Elastic and Automated Pipelining for Fast Distributed Training of Transformer Models")
    parser.add_argument("--nnodes", type=int, default=2)

    parser.add_argument("--nproc_per_node", type=int, default=8)

    parser.add_argument("--node_rank", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=1)

    parser.add_argument("--global_rank", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(processName)s - %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')

    args.epochs = 10
    args.img_size = 224
    args.dataset = "imagenet"
    args.data_dir = "/home/chaoyanghe/sourcecode/dataset/cv/ImageNet"
    # args.dataset = "cifar10"
    # args.data_dir = "./data/cifar10"
    epoch = 0
    data_manager = CVDatasetManager(args)

    data_manager.set_seed(data_manager.seeds[0])

    num_replicas = [2, 2, 2, 2, 4, 4, 8, 8, 16, 16]

    # train_indices_dataloader = data_manager.test_mapping_for_single_worker()node_rank, num_replicas, local_rank
    starting_time = time.time()
    train_indices_dataloader, _ = data_manager.get_data_loader_with_node_rank(epoch=epoch, batch_size=args.batch_size,
                                                                              node_rank=args.node_rank,
                                                                              num_replicas=num_replicas[0],
                                                                              local_rank=args.local_rank)
    # train_indices_dataloader, _ = data_manager.get_data_loader_single_worker(batch_size=batch_size)
    # (global_rank=0, local_rank=1, nnodes=2, node_rank=0, nproc_per_node=8)
    logging.info("len of train_indices_dataloader = %d" % len(train_indices_dataloader))
    ending_time = time.time()
    # logging.info(data_manager.get_train_sample_index(0))
    logging.info("time cost = " + str(ending_time - starting_time))

    hidden_feature_size = 512 * 769 * 197 * 4

    auto_cache = AutoCacheImpl(data_manager)

    # epoch, is_ready, batch_size, hidden_feature_size, processes_num
    auto_cache.reset_status(0, False, args.batch_size, hidden_feature_size, 1)

    model = TestModel()
    trainer = Trainer(auto_cache, model)
    trainer.train()

    logging.info("**************************************************************\n\n\n")

    auto_cache.reset_status(0, False, args.batch_size, hidden_feature_size, 1)

    trainer.train()
