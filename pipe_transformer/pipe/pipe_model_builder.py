import logging
import time

import torch
import torch.nn as nn
from transformers import apply_chunking_to_forward

"""
ViT

Pipeline only support nn.Sequential partitions thus a layer cannot take more than 1 preceding layers as input.
This model architecture is normal in practice, such as Transformer Multi Head Attention layer, and the skip connection in ResNet.

Without such support, the partition can only be coarse-grained, which is hard to make the word load balanced in pipeline.
"""
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, attention_norm, attn):
        super(MultiHeadAttentionLayer, self).__init__()
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


class FrozenLayer(nn.Module):
    def __init__(self, num_frozen_layer, frozen_emb, frozen_layer_list):
        super().__init__()
        self.num_frozen_layer = num_frozen_layer
        self.embedding = frozen_emb
        self.layers = nn.ModuleList()
        for layer_i in range(num_frozen_layer):
            self.layers.append(frozen_layer_list[layer_i])

    def forward(self, x, layer_id=0):
        logging.info(x)
        if layer_id == self.num_frozen_layer:
            logging.info("no need to recompute")
            return x
        if layer_id == 0:
            logging.info("compute from layer 0")
            if isinstance(x, dict):
                # NLP
                x = self.embedding(input_ids=x['input_ids'])
            else:
                # CV
                x = self.embedding(x)
            for id in range(0, self.num_frozen_layer):
                x = self.layers[id](x)
            return x
        else:
            logging.info("compute from layer %d-%d" % (layer_id, self.num_frozen_layer-1))
            for id in range(layer_id, self.num_frozen_layer):
                x = self.layers[id](x)
            return x


"""
For BERT

Pipeline only support nn.Sequential partitions thus a layer cannot take more than 1 preceding layers as input.
This model architecture is normal in practice, such as Transformer Multi Head Attention layer, and the skip connection in ResNet.

Without such support, the partition can only be coarse-grained, which is hard to make the word load balanced in pipeline.
"""


class BertFFNLayer(nn.Module):
    def __init__(self, config, intermediate, output):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = intermediate
        self.output = output

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


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


def count_parameters(model, b_is_required_grad=True):
    if b_is_required_grad:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in model.parameters())
    return params / 1000000


def create_pipe_styled_model_vit(model_backbone, output_model, num_layer_in_total, num_frozen_layer):
    """
    Optimization:
        Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
        Prepare a Pin Memory model
    """
    logging.info(model_backbone)
    frozen_model = None
    pipe_model = nn.Sequential()

    parameters_size_frozen = 0.0
    parameters_list_pipe = []

    if num_frozen_layer > 0:
        for param in model_backbone.transformer.embeddings.parameters():
            param.requires_grad = False

        frozen_emb = model_backbone.transformer.embeddings

        size_embedding = count_parameters(model_backbone.transformer.embeddings, False)
        parameters_size_frozen += size_embedding

        frozen_layer_list = nn.ModuleList()
        for frozen_layer_index in range(num_frozen_layer):
            layer_block = model_backbone.transformer.encoder.layer[frozen_layer_index]
            for param in layer_block.parameters():
                param.requires_grad = False
            frozen_layer_list.append(layer_block)

            size_layer_block = count_parameters(layer_block, False)
            parameters_size_frozen += size_layer_block

        frozen_model = FrozenLayer(num_frozen_layer, frozen_emb, frozen_layer_list)
    else:
        pipe_model.add_module("embedding", model_backbone.transformer.embeddings)
        size_embedding = count_parameters(model_backbone.transformer.embeddings, False)
        parameters_list_pipe.append(size_embedding)

    # add transformer blocks needed to be trained
    for layer_index in range(num_frozen_layer, num_layer_in_total):
        layer_block = model_backbone.transformer.encoder.layer[layer_index]
        multihead_attention_layer = MultiHeadAttentionLayer(layer_block.attention_norm, layer_block.attn)
        mlp_layer = MLPLayer(layer_block.ffn_norm, layer_block.ffn)
        pipe_model.add_module("multihead_attention_layer" + str(layer_index), multihead_attention_layer)
        pipe_model.add_module("mlp_layer" + str(layer_index), mlp_layer)

        size_multihead_attention_layer = count_parameters(multihead_attention_layer, False)
        parameters_list_pipe.append(size_multihead_attention_layer)

        size_mlp_layer = count_parameters(mlp_layer, False)
        parameters_list_pipe.append(size_mlp_layer)

    pipe_model.add_module("encoder_norm", model_backbone.transformer.encoder.encoder_norm)
    size_encoder_norm = count_parameters(model_backbone.transformer.encoder.encoder_norm, False)
    parameters_list_pipe.append(size_encoder_norm)

    pipe_model.add_module("head", output_model)
    size_output_model = count_parameters(output_model, False)
    parameters_list_pipe.append(size_output_model)

    # logging.info(frozen_model)
    # logging.info(parameters_size_frozen)
    # logging.info(pipe_model)
    # logging.info(parameters_list_pipe)

    return frozen_model, parameters_size_frozen, pipe_model, parameters_list_pipe

def create_pipe_styled_model_BERT_for_TC(model_config, model_backbone, output_model, num_layer_in_total, num_frozen_layer):
    logging.info(model_backbone)
    for name, p in model_backbone.named_parameters():
        logging.info(name)

    frozen_model = None
    pipe_model = nn.Sequential()

    parameters_size_frozen = 0.0
    parameters_list_pipe = []

    if num_frozen_layer > 0:
        for param in model_backbone.bert.embeddings.parameters():
            param.requires_grad = False

        frozen_emb = model_backbone.bert.embeddings

        size_embedding = count_parameters(model_backbone.bert.embeddings, False)
        parameters_size_frozen += size_embedding

        frozen_layer_list = nn.ModuleList()
        for frozen_layer_index in range(num_frozen_layer):
            layer_block = model_backbone.bert.encoder.layer[frozen_layer_index]
            for param in layer_block.parameters():
                param.requires_grad = False
            frozen_layer_list.append(layer_block)

            size_layer_block = count_parameters(layer_block, False)
            parameters_size_frozen += size_layer_block

        frozen_model = FrozenLayer(num_frozen_layer, frozen_emb, frozen_layer_list)
    else:
        pipe_model.add_module("embedding", model_backbone.bert.embeddings)
        size_embedding = count_parameters(model_backbone.bert.embeddings, False)
        parameters_list_pipe.append(size_embedding)

    # add transformer blocks needed to be trained
    for layer_index in range(num_frozen_layer, num_layer_in_total):
        layer_block = model_backbone.bert.encoder.layer[layer_index]

        pipe_model.add_module("layer" + str(layer_index) + "attention", layer_block.attention)
        size_layer_block_attention = count_parameters(layer_block.attention, False)
        parameters_list_pipe.append(size_layer_block_attention)
        # logging.info(size_layer_block_attention)

        ffn_layer = BertFFNLayer(model_config, layer_block.intermediate, layer_block.output)

        pipe_model.add_module("layer" + str(layer_index) + "ffn_layer", ffn_layer)
        size_layer_ffn_layer = count_parameters(ffn_layer, False)
        parameters_list_pipe.append(size_layer_ffn_layer)
        # logging.info(size_layer_ffn_layer)

    pipe_model.add_module("pooler", model_backbone.bert.pooler)
    size_pooler = count_parameters(model_backbone.bert.pooler, False)
    parameters_list_pipe.append(size_pooler)

    pipe_model.add_module("classifier", model_backbone.classifier)
    size_classifier = count_parameters(model_backbone.classifier, False)
    parameters_list_pipe.append(size_classifier)


    logging.info(frozen_model)
    logging.info(parameters_size_frozen)
    logging.info(pipe_model)
    logging.info(parameters_list_pipe)

    return frozen_model, parameters_size_frozen, pipe_model, parameters_list_pipe


def create_pipe_styled_model_BERT_for_QA(model_backbone, output_model, num_layer_in_total, num_frozen_layer):
    pass


def create_pipe_styled_model(config, model_config, model_backbone, output_model, num_layer_in_total, num_frozen_layer):
    if config.learning_task == config.LEARNING_TASK_IMAGE_CLASSIFICATION and \
            config.model_name == config.MODEL_VIT:
        logging.info("create ViT pipeline")
        return create_pipe_styled_model_vit(model_backbone, output_model, num_layer_in_total, num_frozen_layer)

    elif config.learning_task == config.LEARNING_TASK_TEXT_CLASSIFICATION and \
            config.model_name == config.MODEL_BERT:
        logging.info("create BERT for text classification pipeline")
        return create_pipe_styled_model_BERT_for_TC(model_config, model_backbone, output_model, num_layer_in_total, num_frozen_layer)

    elif config.learning_task == config.LEARNING_TASK_QUESTION_ANSWERING and \
            config.model_name == config.MODEL_BERT:
        logging.info("create BERT for QA pipeline")
        return create_pipe_styled_model_BERT_for_QA(model_backbone, output_model, num_layer_in_total, num_frozen_layer)
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
    logging.info("input = " + str(pipe))
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
    output_pipe_model =  nn.Sequential(*balanced_pipe)
    logging.info("output = " + str(output_pipe_model))
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
