import logging
import time

import torch
from torch import nn

from .bert_qa_partition import create_pipe_styled_model_BERT_for_QA
from .bert_tc_partition import create_pipe_styled_model_BERT_for_TC
from .vit_partition import create_pipe_styled_model_vit, freeze_vit_only

"""
Issues Description:
the output of pipe is RRef，but DDP cannot recognize RRef object so DDP cannot find Tensor inside RRef.
Then DDP will view all parameters as used ones.

Temporal Solution:
Using a Wrapper model to help DDP find find Tensors inside RRef.
"""


class PipeModelWrapper(nn.Module):
    def __init__(self, pipe_model):
        super().__init__()
        self.pipe_model = pipe_model

    def forward(self, *args, **kwargs):
        return self.pipe_model(*args, **kwargs).local_value()


def create_pipe_styled_model(config, model_config, model_backbone, num_layer_in_total, num_frozen_layer):
    if config.learning_task == config.LEARNING_TASK_IMAGE_CLASSIFICATION and \
            config.model_name == config.MODEL_VIT:
        logging.info("create ViT pipeline")
        return create_pipe_styled_model_vit(model_config, model_backbone, num_layer_in_total, num_frozen_layer)

    elif config.learning_task == config.LEARNING_TASK_TEXT_CLASSIFICATION and \
            config.model_name == config.MODEL_BERT:
        logging.info("create BERT for text classification pipeline")
        return create_pipe_styled_model_BERT_for_TC(model_config, model_backbone, num_layer_in_total, num_frozen_layer)

    elif config.learning_task == config.LEARNING_TASK_QUESTION_ANSWERING and \
            config.model_name == config.MODEL_BERT:
        logging.info("create BERT for QA pipeline")
        return create_pipe_styled_model_BERT_for_QA(model_config, model_backbone, num_layer_in_total,
                                                    num_frozen_layer)
    else:
        raise Exception("does not exist")


def freeze_only(config, model_config, model_backbone, num_layer_in_total, num_frozen_layer):
    if config.learning_task == config.LEARNING_TASK_IMAGE_CLASSIFICATION and \
            config.model_name == config.MODEL_VIT:
        logging.info("create ViT pipeline")
        return freeze_vit_only(model_backbone, num_frozen_layer)

    elif config.learning_task == config.LEARNING_TASK_TEXT_CLASSIFICATION and \
            config.model_name == config.MODEL_BERT:
        logging.info("create BERT for text classification pipeline")

    elif config.learning_task == config.LEARNING_TASK_QUESTION_ANSWERING and \
            config.model_name == config.MODEL_BERT:
        logging.info("create BERT for QA pipeline")
    else:
        raise Exception("does not exist")


def convert_to_balanced_model(local_rank, global_rank,
                              device_idx_start, pipe: nn.Sequential, balance):
    # logging.info("device_idx_start = %d" % device_idx_start)
    # logging.info(pipe)
    # logging.info(balance)
    """
    Optimization:
        Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
        Prepare a Pin Memory model
    """
    # logging.info("input = " + str(pipe))
    logging.info("convert_to_balanced_model. local_rank = %d, global_rank = %d" % (local_rank, global_rank))
    time_start_loading = time.time()
    pipe_layer_idx = 0
    balanced_pipe = []
    for device_id in balance.keys():
        num_layers = balance[device_id]
        layers = []
        for i in range(num_layers):
            layers.append(pipe[pipe_layer_idx])
            pipe_layer_idx += 1
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(device_id + device_idx_start))
            logging.info("######################local_rank = %d, global_rank = %d, device id: %d" % (local_rank,
                                                                                                     global_rank,
                                                                                                     device_id + device_idx_start))
            balanced_pipe.append(nn.Sequential(*layers).to(device, non_blocking=False))
        else:
            balanced_pipe.append(nn.Sequential(*layers))
    time_end_loading = time.time()
    logging.info("CPU->GPU time cost = " + str(time_end_loading - time_start_loading))
    output_pipe_model = nn.Sequential(*balanced_pipe)
    #    logging.info("output = " + str(output_pipe_model))
    return output_pipe_model


def freeze_layers_for_pipe_model(model, num_frozen_layers):
    ddp_ignore_name_list = []
    partition_idx = 0
    sub_layer_idx = 0
    for i in range(num_frozen_layers * 2 + 1):
        # the frozen layers may be split into multiple partitions
        if sub_layer_idx > len(model.partitions[partition_idx]) - 1:
            partition_idx += 1
            sub_layer_idx = 0

        for param in model.partitions[partition_idx][sub_layer_idx].parameters():
            param.requires_grad = False

        sub_layer_idx += 1

    logging.info(ddp_ignore_name_list)
    return ddp_ignore_name_list


def freeze_layers_for_normal_model(model, num_frozen_layers):
    if num_frozen_layers > 0:
        for param in model.transformer.embeddings.parameters():
            param.requires_grad = False
        for frozen_layer_index in range(num_frozen_layers):
            layer_block = model.transformer.encoder.layer[frozen_layer_index]
            for param in layer_block.parameters():
                param.requires_grad = False


def get_ddp_ignored_params_name(model, num_frozen_layers):
    if num_frozen_layers == 0:
        return []

    def get_name_list_to_ignore_comm_in_ddp(model, model_module):
        model_emb_name = [
            module_name
            for module_name, module in model.named_modules()
            if module is model_module
        ][0]
        proxy_param_names = [
            f"{model_emb_name}.{param_name}"
            for param_name, _ in model_module.named_parameters()
        ]
        proxy_buffer_names = [
            f"{model_emb_name}.{buf_name}"
            for buf_name, _ in model_module.named_buffers()
        ]
        return proxy_param_names + proxy_buffer_names

    ddp_ignore_name_list = []
    partition_idx = 0
    sub_layer_idx = 0
    for i in range(num_frozen_layers * 2 + 1):
        # the frozen layers may be split into multiple partitions
        if sub_layer_idx > len(model.partitions[partition_idx]) - 1:
            partition_idx += 1
            sub_layer_idx = 0

        name_list = get_name_list_to_ignore_comm_in_ddp(model, model.partitions[partition_idx][sub_layer_idx])
        # logging.info(name_list)
        ddp_ignore_name_list += name_list

        sub_layer_idx += 1

    logging.info(ddp_ignore_name_list)
    return ddp_ignore_name_list
