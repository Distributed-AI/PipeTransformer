import torch
from torch.distributed.pipeline.sync import Pipe

from pipe.load_balance import generate_parameter_size_wise_balance
from pipe.pipe_model_builder import convert_to_balanced_model, create_pipe_styled_model, get_ddp_ignored_params_name, \
    freeze_layers_for_pipe_model, PipeModelWrapper

"""
Under development by Chaoyang He
    
Chaoyang's idea for elastic pipe：
1. 首先按照8卡去创建单个pipe，然后均衡分配所有的layer，也能确定单个device能使用的最大batch size (fixed during training) 和max_layer_per_device的值
2. 如果有frozen，需要训练的layer数足够分配给所有num_gpu_device*max_layer_per_device时，就只是重新改变partition的分布使得更为均衡，称之为elastic partition
3. 如果继续frozen，需要训练的layer数小于了num_gpu_device*max_layer_per_device，也就是不够分配了，会创建新的pipe，缩小单个pipe的device数，然后按照2的算法做elastic partition。pipe增加的过程称之为elastic pipe generation
4. 最后继续frozen到只有max_layer_per_device需要训练的时候，直接降级为ddp，所有pipe销毁，也就是每个device创建一个进程，进行ddp。这个步骤称之为extreme pipe generation (名字暂定)

"""


class AutoElasticPipe:
    def __init__(self, world_size, local_rank, global_rank, model_backbone, output_head, num_device, num_layer_in_total,
                 debug_mode=False):
        self.world_size = world_size
        self.local_rank = local_rank
        self.global_rank = global_rank

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

    def transform(self, num_frozen_layers):
        # traceback.print_stack()
        print("---local_rank = %d, global_rank = %d -------------freeze layer number = %d---------------" % (
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
        print("len(pipe_model) = %d" % len(model))
        print("len(pipe_model paras_size) = %d" % len(self.pipe_model_params_size_list))

        if num_frozen_layers == 0:
            # set the num_frozen_layers = 0 because we put all frozen layers into frozen_model
            balanced_sub_layer_distribution, balanced_params_size_distribution = self._auto_balanced_elastic_partition(0)
            self.max_parameter_per_gpu_at_beginning = max(balanced_params_size_distribution.values())
            print("self.max_parameter_per_gpu_at_beginning = %f" % self.max_parameter_per_gpu_at_beginning)
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

    def get_num_frozen_layers(self):
        return self.num_frozen_layers

    def get_pipe_len(self):
        return self.pipe_len

    def set_pipe_len(self, pipe_len):
        self.pipe_len = pipe_len

    def get_device_first(self):
        device_first = torch.device("cuda:" + str(self.local_rank * self.pipe_len))
        print(device_first)
        return device_first

    def get_device_last(self):
        device_last = torch.device("cuda:" + str((self.local_rank + 1) * self.pipe_len - 1))
        print(device_last)
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

        print(balanced_sub_layer_distribution)
        print(balanced_params_size_distribution)
        return balanced_sub_layer_distribution, balanced_params_size_distribution

    def _auto_pipe_length(self, num_frozen_layers):
        # split the model into a pipe sequential
        # Note if the pipe len = 1 and the batch size is big, we cannot hold the model in a single GPU
        # thus we keep the minimum pipe len as 2.
        # another strategy is using smaller batch size, but this may lead to accuracy drop and convergence uncertainty

        # 8 (pipe) -> 4 (pipe) -> 2 (pipe) -> 1(DP)
        while self.pipe_len >= 2:
            # detect the max parameter size per GPU after shrink device number
            print("----------start to detection---------")
            balanced_sub_layer_distribution, balanced_params_size_distribution, self.frozen_params = generate_parameter_size_wise_balance(
                int(self.pipe_len / 2),
                self.pipe_model_params_size_list,
                num_frozen_layers)
            balanced_params_size_distribution[0] -= self.frozen_params * (5.0 / 6.0)
            max_parameter_per_gpu = max(balanced_params_size_distribution.values())
            print("max_parameter_per_gpu = %f" % max_parameter_per_gpu)
            print("self.max_parameter_per_gpu_at_beginning = %f" % self.max_parameter_per_gpu_at_beginning)
            if max_parameter_per_gpu <= self.max_parameter_per_gpu_at_beginning:
                print("#########    add pipe    #######")
                self.pipe_len = int(self.pipe_len / 2)
            else:
                break

        print("current_num_device = %d" % self.pipe_len)

    def _get_pipe(self, model):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.pipe = Pipe(model, chunks=6, checkpoint="never")
        return self.pipe
