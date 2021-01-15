import logging

import torch

from pipe import Pipe
from pipe.load_balance import generate_parameter_size_wise_balance
from pipe.pipe_model_builder import convert_to_balanced_model, create_pipe_styled_model, PipeModelWrapper


class AutoElasticPipe:
    def __init__(self, world_size, local_rank, global_rank, num_chunks_of_micro_batches, model_backbone, output_head, num_device, num_layer_in_total,
                 debug_mode=False):
        self.world_size = world_size
        self.local_rank = local_rank
        self.global_rank = global_rank

        self.num_chunks_of_micro_batches = num_chunks_of_micro_batches

        self.model_backbone = model_backbone
        self.output_head = output_head
        self.normal_model = model_backbone
        """
        Optimization:
            Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
            Prepare a Pin Memory model
        """
        # if torch.cuda.is_available():
        #     for p in self.model_backbone.parameters():
        #         p.pin_memory()
        #     for p in self.output_head.parameters():
        #         p.pin_memory()
        #     for p in self.normal_model.parameters():
        #         p.pin_memory()

        self.num_device_at_beginning = num_device
        self.pipe_len = num_device
        self.num_layer_in_total = num_layer_in_total

        self.debug_mode = debug_mode

        # pipe
        self.pipe = None
        self.pipe_model_params_size_list = []
        self.frozen_params = 0.0
        self.max_parameter_per_gpu_at_beginning = 0.0
        self.num_frozen_layers = -1

        # switch
        self.b_enable = True

    def enable(self, on):
        self.b_enable = on

    def transform(self, num_frozen_layers):
        # traceback.print_stack()
        logging.info("---local_rank = %d, global_rank = %d -------------freeze layer number = %d---------------" % (
            self.local_rank,
            self.global_rank,
            num_frozen_layers))
        # if self.is_first_call and num_frozen_layers != 0:
        #     raise Exception("the first transformation must from all layers training")

        self.num_frozen_layers = num_frozen_layers
        # frozen_model, parameters_size_frozen, pipe_model, parameters_list_pipe

        frozen_model, parameters_size_frozen, \
        model, self.pipe_model_params_size_list = create_pipe_styled_model(self.model_backbone, self.output_head,
                                                                           self.num_layer_in_total, num_frozen_layers)
        logging.info("len(pipe_model) = %d" % len(model))
        logging.info("len(pipe_model paras_size) = %d" % len(self.pipe_model_params_size_list))

        # when b_enable = False, the load balance is not even, may lead to lower training speed.
        if num_frozen_layers == 0 or not self.b_enable:
            # set the num_frozen_layers = 0 because we put all frozen layers into frozen_model
            balanced_sub_layer_distribution, balanced_params_size_distribution = self._auto_balanced_elastic_partition(
                0)
            self.max_parameter_per_gpu_at_beginning = max(balanced_params_size_distribution.values())
            logging.info("self.max_parameter_per_gpu_at_beginning = %f" % self.max_parameter_per_gpu_at_beginning)
        else:
            self._auto_pipe_length(num_frozen_layers)
            # set the num_frozen_layers = 0 because we put all frozen layers into frozen_model
            balanced_sub_layer_distribution, _ = self._auto_balanced_elastic_partition(0)

        device_idx_start = self.local_rank * self.pipe_len
        model = convert_to_balanced_model(self.local_rank, self.global_rank,
                                          device_idx_start, model, balanced_sub_layer_distribution)
        # frozen model is always in device 0
        if frozen_model is not None:
            frozen_model.to(device_idx_start)

        pipe_model = self._get_pipe(model)

        # params_to_skip = get_ddp_ignored_params_name(pipe_model, num_frozen_layers)

        return frozen_model, PipeModelWrapper(pipe_model), self.pipe_len

    def get_origin_model(self):
        return self.model_backbone

    def get_num_frozen_layers(self):
        return self.num_frozen_layers

    def get_pipe_len(self):
        return self.pipe_len

    def set_pipe_len(self, pipe_len):
        self.pipe_len = pipe_len

    def get_device_first(self):
        device_first = torch.device("cuda:" + str(self.local_rank * self.pipe_len))
        logging.info(device_first)
        return device_first

    def get_device_last(self):
        device_last = torch.device("cuda:" + str((self.local_rank + 1) * self.pipe_len - 1))
        logging.info(device_last)
        return device_last

    def get_max_parameter_per_gpu_at_beginning(self):
        return self.max_parameter_per_gpu_at_beginning

    def set_max_parameter_per_gpu_at_beginning(self, max_parameter_per_gpu_at_beginning):
        self.max_parameter_per_gpu_at_beginning = max_parameter_per_gpu_at_beginning

    def _auto_balanced_elastic_partition(self, num_frozen_layers):
        balanced_sub_layer_distribution, balanced_params_size_distribution, self.frozen_params = generate_parameter_size_wise_balance(
            self.pipe_len,
            self.pipe_model_params_size_list,
            num_frozen_layers)

        logging.info(balanced_sub_layer_distribution)
        logging.info(balanced_params_size_distribution)
        return balanced_sub_layer_distribution, balanced_params_size_distribution

    def _auto_pipe_length(self, num_frozen_layers):
        # split the model into a pipe sequential
        # Note if the pipe len = 1 and the batch size is big, we cannot hold the model in a single GPU
        # thus we keep the minimum pipe len as 2.
        # another strategy is using smaller batch size, but this may lead to accuracy drop and convergence uncertainty

        # 8 (pipe) -> 4 (pipe) -> 2 (pipe) -> 1(DP)
        while self.pipe_len >= 2:
            # detect the max parameter size per GPU after shrink device number
            logging.info("----------start to detection---------")
            balanced_sub_layer_distribution, balanced_params_size_distribution, self.frozen_params = generate_parameter_size_wise_balance(
                int(self.pipe_len / 2),
                self.pipe_model_params_size_list, 0)
            balanced_params_size_distribution[0] -= self.frozen_params * (5.0 / 6.0)
            max_parameter_per_gpu = max(balanced_params_size_distribution.values())
            logging.info("max_parameter_per_gpu = %f" % max_parameter_per_gpu)
            logging.info("self.max_parameter_per_gpu_at_beginning = %f" % self.max_parameter_per_gpu_at_beginning)
            if max_parameter_per_gpu <= self.max_parameter_per_gpu_at_beginning:
                logging.info("#########    add pipe    #######")
                self.pipe_len = int(self.pipe_len / 2)
            else:
                break

        logging.info("current_num_device = %d" % self.pipe_len)

    def _get_pipe(self, model):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        # self.num_chunks_of_micro_batches
        self.pipe = Pipe(model, chunks=self.num_chunks_of_micro_batches, checkpoint="never")
        return self.pipe

    def _get_optimal_chunk_num_by_pipe_len(self, pipe_len):
        if pipe_len == 8:
            chunk_num = 4 * pipe_len
        elif pipe_len == 4:
            chunk_num = 4 * pipe_len
        elif pipe_len == 2:
            chunk_num = 4 * pipe_len
        else:
            chunk_num = 4 * pipe_len
        return chunk_num