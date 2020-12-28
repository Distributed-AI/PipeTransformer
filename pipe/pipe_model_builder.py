import time

import torch
import torch.nn as nn


class MultHeadAttentionLayer(nn.Module):
    def __init__(self, attention_norm, attn):
        super(MultHeadAttentionLayer, self).__init__()
        self.attention_norm = attention_norm
        self.attn = attn

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        return x


class MLPLayer(nn.Module):
    def __init__(self, ffn_norm, ffn):
        super(MLPLayer, self).__init__()
        self.ffn_norm = ffn_norm
        self.ffn = ffn

    def forward(self, x):
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class OutputHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(OutputHead, self).__init__()
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        logits = self.head(x[:, 0])
        return logits


"""
Issues Description:
the output of pipe is RRef，but DDP cannot recognize RRef object so DDP cannot find Tensor inside RRef.
Then DDP will view all parameters as used ones.

Temporal Solution:
Using a Wrapper model to help DDP find find Tensors inside RRef.
"""


class Wrapper(nn.Module):
    ALL_LAYER = 0
    FROZEN_LAYER = 1
    ACTIVE_LYAER = 2

    def __init__(self, pipe_model, num_frozen_layers):
        super().__init__()
        self.pipe_model = pipe_model
        # print(self.pipe_model)
        self.num_frozen_layers = num_frozen_layers

        self.mode = Wrapper.ALL_LAYER

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, input):
        if self.mode == Wrapper.ALL_LAYER:
            return self.forward_all_layers(input)
        elif self.mode == Wrapper.FROZEN_LAYER:
            return self.forward_frozen_layers(input)
        elif self.mode == Wrapper.ACTIVE_LYAER:
            return self.forward_active_layer(input)
        else:
            raise Exception("cannot work in this mode")

    def forward_all_layers(self, input):
        return self.pipe_model(input).local_value()

    def forward_frozen_layers(self, input):
        hidden_output = input
        layer_idx_in_total = 0

        for partition_idx in range(len(self.pipe_model.partitions)):
            model_partition = self.pipe_model.partitions[partition_idx]
            for sub_layer_idx in range(len(model_partition)):
                model_sub_layer = model_partition[sub_layer_idx]
                hidden_output = model_sub_layer(hidden_output)
                layer_idx_in_total += 1
                if layer_idx_in_total >= self.num_frozen_layers * 2 + 1:
                    return hidden_output.local_value()

    def forward_active_layer(self, input):
        hidden_output = input
        layer_idx_in_total = 0
        is_active_layer = False

        for partition_idx in range(len(self.pipe_model.partitions)):
            model_partition = self.pipe_model.partitions[partition_idx]
            for sub_layer_idx in range(len(model_partition)):
                model_sub_layer = model_partition[sub_layer_idx]
                if is_active_layer:
                    hidden_output = model_sub_layer(hidden_output)
                layer_idx_in_total += 1
                if layer_idx_in_total >= self.num_frozen_layers * 2 + 1:
                    is_active_layer = True
        return hidden_output.local_value()


def count_parameters(model, b_is_required_grad=True):
    if b_is_required_grad:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in model.parameters())
    return params / 1000000


def create_pipe_styled_model(model_backbone, output_model, num_layer_in_total, num_frozen_layer):
    """
    Optimization:
        Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
        Prepare a Pin Memory model
    """
    pipe_model = nn.Sequential()

    parameters_list = []

    if num_frozen_layer > 0:
        for param in model_backbone.transformer.embeddings.parameters():
            param.requires_grad = False

        for frozen_layer_index in range(num_frozen_layer):
            layer_block = model_backbone.transformer.encoder.layer[frozen_layer_index]
            for param in layer_block.parameters():
                param.requires_grad = False

    pipe_model.add_module("embedding", model_backbone.transformer.embeddings)
    size_embedding = count_parameters(model_backbone.transformer.embeddings, False)
    parameters_list.append(size_embedding)

    # add transformer blocks needed to be trained
    for layer_index in range(0, num_layer_in_total):
        layer_block = model_backbone.transformer.encoder.layer[layer_index]
        multihead_attention_layer = MultHeadAttentionLayer(layer_block.attention_norm, layer_block.attn)
        mlp_layer = MLPLayer(layer_block.ffn_norm, layer_block.ffn)
        pipe_model.add_module("multihead_attention_layer" + str(layer_index), multihead_attention_layer)
        pipe_model.add_module("mlp_layer" + str(layer_index), mlp_layer)

        size_multihead_attention_layer = count_parameters(multihead_attention_layer, False)
        parameters_list.append(size_multihead_attention_layer)

        size_mlp_layer = count_parameters(mlp_layer, False)
        parameters_list.append(size_mlp_layer)

    pipe_model.add_module("encoder_norm", model_backbone.transformer.encoder.encoder_norm)
    size_encoder_norm = count_parameters(model_backbone.transformer.encoder.encoder_norm, False)
    parameters_list.append(size_encoder_norm)

    pipe_model.add_module("head", output_model)
    size_output_model = count_parameters(output_model, False)
    parameters_list.append(size_output_model)

    print(parameters_list)

    return pipe_model, parameters_list


def convert_to_balanced_model(local_rank, global_rank,
                              device_idx_start, pipe: nn.Sequential, balance):
    # print("device_idx_start = %d" % device_idx_start)
    # print(pipe)
    # print(balance)
    """
    Optimization:
        Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
        Prepare a Pin Memory model
    """
    print("convert_to_balanced_model. local_rank = %d, global_rank = %d" % (local_rank, global_rank))
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
            print("######################local_rank = %d, global_rank = %d, device id: %d" % (local_rank,
                                                                                              global_rank,
                                                                                              device_id + device_idx_start))
            balanced_pipe.append(nn.Sequential(*layers).to(device, non_blocking=True))
        else:
            balanced_pipe.append(nn.Sequential(*layers))
    time_end_loading = time.time()
    print("CPU->GPU time cost = " + str(time_end_loading - time_start_loading))
    return nn.Sequential(*balanced_pipe)


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

    print(ddp_ignore_name_list)
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
        # print(name_list)
        ddp_ignore_name_list += name_list

        sub_layer_idx += 1

    print(ddp_ignore_name_list)
    return ddp_ignore_name_list
