import gc
import logging
import os

import torch
import torch.distributed as dist
import torch.nn
from torch.distributed import rpc, timedelta, Backend
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP
from art import *

from .distributed_communicator import dist_broadcast
from ..pipe.model_partition.pipe_model_builder import get_ddp_ignored_params_name


class AutoDataParallel:

    def __init__(self, config):
        self.config = config

        self.enable_new_pipe = config.b_auto_dp

        self.num_nodes = config.num_nodes
        self.local_rank = -1
        self.global_rank = -1
        self.world_size = -1
        self.active_process_group = None

        self.initial_pipe_len = config.pipe_len_at_the_beginning
        self.compressed_pipe_len = config.pipe_len_at_the_beginning

        self.first_run = True

        self.active_ranks = []
        self.newly_added_active_ranks = []
        # key: rank; value: data_rank
        self.active_data_ranks = dict()
        self.freeze_point = None
        self.message_len = 50

        self.comm_broadcast_group = None

        self.init_ddp()
        self.init_rpc()

    def is_enable(self):
        return self.enable_new_pipe

    def get_ddp_model(self, model, local_rank):
        return DDP(model, device_ids=[local_rank], output_device=local_rank)

    def init_ddp(self):
        self.local_rank = self.config.local_rank
        logging.info(f"Running DP on local rank {self.local_rank}.")

        self.config.master_port += 1
        os.environ.update({"MASTER_ADDR": self.config.master_addr})
        os.environ.update({"MASTER_PORT": str(self.config.master_port)})

        # use InfiniBand
        # os.environ['NCCL_DEBUG'] = 'INFO'

        if self.config.is_infiniband:
            os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
            os.environ['GLOO_SOCKET_IFNAME'] = 'ib0'
            os.environ['TP_SOCKET_IFNAME'] = 'ib0'
        else:
            os.environ['NCCL_IB_DISABLE'] = '1'
            os.environ['NCCL_TREE_THRESHOLD'] = '0'
            os.environ['NCCL_SOCKET_IFNAME'] = self.config.if_name
            os.environ['GLOO_SOCKET_IFNAME'] = self.config.if_name
            os.environ['TP_SOCKET_IFNAME'] = self.config.if_name

        # This the global rank: 0, 1, 2, ..., 15
        self.global_rank = int(os.environ['RANK'])
        logging.info("int(os.environ['RANK']) = %d" % self.global_rank)

        # This the globak world_size
        self.world_size = int(os.environ['WORLD_SIZE'])
        logging.info("world_size = %d" % self.world_size)

        # initialize the process group (this must be GLOO)
        dist.init_process_group(init_method='tcp://' + str(self.config.master_addr) + ':' + str(self.config.master_port),
                                backend=Backend.GLOO, rank=self.global_rank, world_size=self.world_size)
        logging.info("init_process_group. local_rank = %d, global_rank = %d" % (self.local_rank, self.global_rank))

    def get_local_rank(self):
        return self.local_rank

    def get_global_rank(self):
        return self.global_rank

    def get_world_size(self):
        return self.world_size

    def get_active_world_size(self):
        return len(self.active_ranks)

    def get_global_data_rank(self):
        self.update_active_ranks()
        logging.info("self.active_data_ranks = " + str(self.active_data_ranks))
        return self.active_data_ranks[self.global_rank]

    def get_global_data_duplicate_num(self):
        self.update_active_ranks()
        return len(self.active_data_ranks)

    def get_local_data_duplicate_num(self):
        return int(len(self.active_ranks)/self.num_nodes)

    def init_rpc(self):
        rpc_backend_options = TensorPipeRpcBackendOptions()
        rpc_backend_options.init_method = 'tcp://' + self.config.master_addr + ':10000'
        rpc.init_rpc(
            "worker:" + str(self.global_rank),
            rank=self.global_rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )
        logging.info("init_rpc")

    def warm_up(self):
        class WarmupModel(torch.nn.Module):
            def __init__(self):
                super(WarmupModel, self).__init__()
                self.model_arch = torch.nn.Parameter(1e-3 * torch.randn(1, 1))

            def forward(self, x):
                x = self.model_arch * 2
                return x

        warmup_model = WarmupModel()
        warmup_model.to(self.local_rank)
        logging.info("local_rank = %d, global_rank = %d" % (self.local_rank, self.global_rank))

    def is_active(self):
        return True if self.global_rank in self.active_ranks else False

    def get_active_ranks(self):
        return self.active_ranks

    def update_active_ranks(self):
        # update active ranks
        new_active_ranks = []
        data_rank = 0
        pipe_num = int(self.initial_pipe_len / self.compressed_pipe_len)
        if self.world_size < self.initial_pipe_len:
            raise Exception("world_size should be divided by self.initial_pipe_len")
        else:
            ddp_num = int(self.world_size / self.initial_pipe_len)
        for dp_idx in range(ddp_num):
            start_rank = dp_idx * self.initial_pipe_len
            for active_rank in range(pipe_num):
                active_rank += start_rank
                new_active_ranks.append(active_rank)
                self.active_data_ranks[active_rank] = data_rank
                data_rank += 1
        logging.info("active ranks = " + str(self.active_ranks))
        self.newly_added_active_ranks = self._diff_list(new_active_ranks, self.active_ranks)
        self.active_ranks.clear()
        self.active_ranks = new_active_ranks

    def get_newly_added_active_ranks(self):
        return self.newly_added_active_ranks

    def create_active_process_group(self):
        if self.active_process_group is None:
            del self.active_process_group
        self.update_active_ranks()
        logging.info("get_active_process_group - auto_pipe.get_active_ranks() = " + str(self.active_ranks))
        logging.info("local_rank = %d, global_rank = %d - *************************create_active_process_group*********"
                     % (self.local_rank, self.global_rank))
        self.active_process_group = dist.new_group(ranks=self.active_ranks, backend=Backend.NCCL,
                                                   timeout=timedelta(days=365))

    def create_broadcast_process_group(self):
        logging.info("create_broadcast_process_group - auto_pipe.get_active_ranks() = " + str(self.active_ranks))
        logging.info(
            "local_rank = %d, global_rank = %d - *************************create_broadcast_process_group*********"
            % (self.local_rank, self.global_rank))
        self.comm_broadcast_group = dist.new_group(ranks=[i for i in range(self.world_size)], backend=Backend.GLOO,
                                                   timeout=timedelta(days=365))

    def generate_ddp_model(self, model, gpu_num_per_process, num_frozen_layers):
        self.pipe_len = gpu_num_per_process
        # ddp_params_to_skip = get_ddp_ignored_params_name(model, num_frozen_layers)
        # DDP._set_params_and_buffers_to_ignore_for_model(model, ddp_params_to_skip)
        if gpu_num_per_process > 1:
            # find_unused_parameters = True can avoid bucket rebuilt, which takes around 20s
            model = DDP(model, process_group=self.active_process_group,
                        find_unused_parameters=True)
            # model = DDP(Wrapper(model), process_group=self.active_process_group)
        else:
            # find_unused_parameters = True can avoid bucket rebuilt, which takes around 20s
            model = DDP(model, device_ids=[self.local_rank], process_group=self.active_process_group,
                        find_unused_parameters=True)
            # model = DDP(Wrapper(model), device_ids=[self.local_rank], process_group=self.active_process_group,
            #             find_unused_parameters=True)
        return model

    def get_freeze_point(self):
        return self.freeze_point

    def transform(self, auto_pipe, auto_freeze, frozen_model, pipe_model, num_frozen_layers, freeze_point):
        self.freeze_point = freeze_point
        if auto_pipe.get_num_frozen_layers() == num_frozen_layers:
            # make sure the AutoCache can reuse the preceding epoch's cache
            logging.info(
                "(%d)!!! no need to transform since auto_pipe.get_num_frozen_layers() == num_frozen_layers" % self.global_rank)
            # raise Exception("!!! no need to transform since auto_pipe.get_num_frozen_layers() == num_frozen_layers")
            return frozen_model, pipe_model, False, False

        if num_frozen_layers == 0 and auto_pipe.get_pipe_len() == 1:
            raise Exception("unexpected trace")

        # create the initial group only once
        if self.first_run:
            self.create_active_process_group()
            self.create_broadcast_process_group()
            self.clear_memory()
            self.first_run = False

        if self.is_active():
            frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed = self._active_process_impl(
                auto_pipe, auto_freeze, num_frozen_layers)
        else:
            frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed = self._inactive_process_impl(
                auto_pipe, auto_freeze)
        return frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed

    def _active_process_impl(self, auto_pipe, auto_freeze, num_frozen_layers):
        is_pipe_len_changed = False
        is_frozen_layer_changed = False
        if auto_pipe.get_num_frozen_layers() != num_frozen_layers:
            is_frozen_layer_changed = True
        frozen_model, pipe_model, pipe_len = auto_pipe.transform(num_frozen_layers)
        if not self.enable_new_pipe:
            self.compressed_pipe_len = pipe_len
            # broadcast control messages
            logging.info("####### broad cast control message (num_frozen_layers, pipe_len) to all processes #######")
            self.clear_memory()
            is_pipe_len_changed = True
            pipe_model = self.generate_ddp_model(pipe_model, pipe_len, num_frozen_layers)
            return frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed

        if self.compressed_pipe_len != pipe_len:
            self.compressed_pipe_len = pipe_len
            self.update_active_ranks()

            # save model weights for newly active processes
            if self.get_local_rank() == 0:
                torch.save(frozen_model.state_dict(), self._build_path_for_frozen_layer_model_state(num_frozen_layers))

            # broadcast control messages
            logging.info("####### broad cast control message (num_frozen_layers, pipe_len) to all processes #######")
            max_parameter_per_gpu_at_beginning = auto_pipe.get_max_parameter_per_gpu_at_beginning()
            num_freeze_layers, last_grad_norm_by_layer = auto_freeze.get_status()

            broad_cast_msg = self._build_broad_cast_message(num_frozen_layers, pipe_len,
                                                            max_parameter_per_gpu_at_beginning, self.freeze_point,
                                                            last_grad_norm_by_layer)
            if self.global_rank == 0:
                logging.info("local_rank = %d, global_rank = %d - *************************dist_send send(START): %s"
                             % (self.local_rank, self.global_rank, str(broad_cast_msg)))
                dist_broadcast(broad_cast_msg, 0, self.comm_broadcast_group)
                logging.info("local_rank = %d, global_rank = %d - *************************dist_send send(END)"
                             % (self.local_rank, self.global_rank))
            else:
                dist_broadcast(broad_cast_msg, 0, self.comm_broadcast_group)

            self.create_active_process_group()
            self.clear_memory()
            is_pipe_len_changed = True
        pipe_model = self.generate_ddp_model(pipe_model, pipe_len, num_frozen_layers)
        return frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed

    def _inactive_process_impl(self, auto_pipe, auto_freeze):

        broad_cast_msg = [float(i * 0.0) for i in range(self.message_len)]
        frozen_message = dist_broadcast(broad_cast_msg, 0, self.comm_broadcast_group)

        num_frozen_layers, pipe_len, max_parameter_per_gpu_at_beginning, newly_added_active_ranks, freeze_point, \
        last_grad_norm_by_layer = self._parse_broad_cast_message(frozen_message)

        is_pipe_len_changed = True
        is_frozen_layer_changed = True
        self.compressed_pipe_len = pipe_len
        auto_pipe.set_pipe_len(pipe_len)
        auto_pipe.set_max_parameter_per_gpu_at_beginning(max_parameter_per_gpu_at_beginning)
        auto_freeze.update_status(num_frozen_layers, last_grad_norm_by_layer)

        self.create_active_process_group()
        self.clear_memory()

        if self.global_rank in newly_added_active_ranks:
            logging.info("global_rank %d is activated!" % self.global_rank)

            frozen_model, pipe_model, pipe_len = auto_pipe.transform(num_frozen_layers)
            pipe_model = self.generate_ddp_model(pipe_model, pipe_len)

            # load model state for frozen_layer
            frozen_model.load_state_dict(torch.load(self._build_path_for_frozen_layer_model_state(num_frozen_layers)))
        else:
            frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed = self._inactive_process_impl(auto_pipe, auto_freeze)
        return frozen_model, pipe_model, is_pipe_len_changed, is_frozen_layer_changed

    def _build_broad_cast_message(self, num_frozen_layers, pipe_len, max_parameter_per_gpu_at_beginning,
                                  freeze_point, last_grad_norm_by_layer):
        logging.info("self.newly_added_active_ranks = " + str(self.newly_added_active_ranks))
        broad_cast_msg = [float(i * 0.0) for i in range(self.message_len)]
        broad_cast_msg[0] = float(num_frozen_layers)
        broad_cast_msg[1] = float(pipe_len)
        broad_cast_msg[2] = float(max_parameter_per_gpu_at_beginning)
        broad_cast_msg[3] = float(freeze_point['epoch'])
        broad_cast_msg[4] = float(len(self.newly_added_active_ranks))
        for idx, new_active_rank in enumerate(self.newly_added_active_ranks):
            broad_cast_msg[idx + 5] = float(new_active_rank)
        if last_grad_norm_by_layer is not None:
            for layer_idx in last_grad_norm_by_layer.keys():
                broad_cast_msg[layer_idx + 5 + len(self.newly_added_active_ranks)] = last_grad_norm_by_layer[layer_idx]
        else:
            broad_cast_msg[5 + len(self.newly_added_active_ranks)] = -1

        art = text2art("PipeTransformer!")
        logging.critical("\n%s" % art)
        logging.critical("\n################################ Congratulations! To train faster, "
                     "PipeTransformer has automatically transformed to:\n"
                     "################################ Epoch: %d \n"
                     "################################ Number of frozen layers: %d \n"
                     "################################ Pipe length: %d/%d \n"
                     "################################ Newly added ranks: %s \n"
                     % (freeze_point['epoch'], num_frozen_layers, pipe_len,
                        self.initial_pipe_len, str(self.newly_added_active_ranks)))

        return broad_cast_msg

    def _parse_broad_cast_message(self, frozen_message):
        logging.info("local_rank = %d, global_rank = %d - frozen_message = %s" % (
            self.local_rank, self.global_rank, frozen_message))

        num_frozen_layers = int(frozen_message[0])
        logging.info("_parse_broad_cast_message. num_frozen_layers = %d" % num_frozen_layers)
        pipe_len = int(frozen_message[1])
        max_parameter_per_gpu_at_beginning = float(frozen_message[2])
        logging.info("local_rank = %d, global_rank = %d - frozen_layer_num = %s" % (
            self.local_rank, self.global_rank, num_frozen_layers))
        logging.info(
            "local_rank = %d, global_rank = %d - pipe_len = %s" % (self.local_rank, self.global_rank, pipe_len))

        epoch_start = int(frozen_message[3])
        freeze_point = dict()
        freeze_point['epoch'] = epoch_start
        self.freeze_point = freeze_point

        size_of_newly_added_active_ranks = int(frozen_message[4])
        newly_added_active_ranks = []
        for i in range(size_of_newly_added_active_ranks):
            newly_added_active_ranks.append(int(frozen_message[i + 5]))
        logging.info("newly_added_active_ranks = " + str(newly_added_active_ranks))

        if frozen_message[5 + size_of_newly_added_active_ranks] != -1:
            last_grad_norm_by_layer = dict()
            for layer_idx in range(self.config.num_layer):
                last_grad_norm_by_layer[layer_idx] = float(frozen_message[layer_idx + 5 + size_of_newly_added_active_ranks])
            logging.info("last_grad_norm_by_layer = " + str(last_grad_norm_by_layer))
        else:
            last_grad_norm_by_layer = None
        return num_frozen_layers, pipe_len, max_parameter_per_gpu_at_beginning, \
               newly_added_active_ranks, freeze_point, last_grad_norm_by_layer

    def observe_params_communicated(self, model):
        def my_hook(state, bucket):
            params_size = sum([t.numel() for t in bucket.get_tensors()])
            global size_params_ddp_sum
            size_params_ddp_sum += params_size
            logging.info(params_size / 1000 / 1000)
            fut = torch.futures.Future()
            fut.set_result(bucket.get_tensors())
            return fut

        model.register_comm_hook(state=None, hook=my_hook)

    def clear_memory(self):
        # dist.destroy_process_group()
        torch.cuda.empty_cache()
        gc.collect()

    def cleanup(self):
        rpc.shutdown(graceful=False)

    def _diff_list(self, li1, li2):
        return (list(list(set(li1) - set(li2)) + list(set(li2) - set(li1))))

    def _build_path_for_frozen_layer_model_state(self, number_frozen_layers):
        return "./checkpoint_frozen_layers.pth_num_frozen_layers_%d.pth" % number_frozen_layers
