import logging

from torch import nn

from .utils import count_parameters

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


class ViTOutputHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ViTOutputHead, self).__init__()
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
            x = self.embedding(x)
            for id in range(0, self.num_frozen_layer):
                x = self.layers[id](x)
            return x
        else:
            logging.info("compute from layer %d-%d" % (layer_id, self.num_frozen_layer - 1))
            for id in range(layer_id, self.num_frozen_layer):
                x = self.layers[id](x)
            return x


def create_pipe_styled_model_vit(model_config, model_backbone, num_layer_in_total, num_frozen_layer):
    """
    Optimization:
        Pin Memory: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
        Prepare a Pin Memory model
    """
    #    logging.info(model_backbone)
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

    output_head = ViTOutputHead(model_config.hidden_size, model_config.output_dim)
    pipe_model.add_module("head", output_head)
    size_output_model = count_parameters(output_head, False)
    parameters_list_pipe.append(size_output_model)

    # logging.info(frozen_model)
    # logging.info(parameters_size_frozen)
    # logging.info(pipe_model)
    # logging.info(parameters_list_pipe)

    return frozen_model, parameters_size_frozen, pipe_model, parameters_list_pipe
