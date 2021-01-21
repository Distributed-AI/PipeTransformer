import logging

from torch import nn

from transformers import apply_chunking_to_forward

"""
For BERT + QA

Pipeline only support nn.Sequential partitions thus a layer cannot take more than 1 preceding layers as input.
This model architecture is normal in practice, such as Transformer Multi Head Attention layer, and the skip connection in ResNet.

Without such support, the partition can only be coarse-grained, which is hard to make the word load balanced in pipeline.
"""


def count_parameters(model, b_is_required_grad=True):
    if b_is_required_grad:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in model.parameters())
    return params / 1000000


class BertFFNLayerForQA(nn.Module):
    def __init__(self, config, intermediate, output):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.intermediate = intermediate
        self.output = output

    def forward(self, self_attention_outputs):
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # outputs = (layer_output,) + outputs
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertForQA_OutputHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        logging.info("config.num_labels = %d" % config.num_labels)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, bert_outputs):
        logits = self.qa_outputs(bert_outputs)
        return logits


class BertFrozenLayerForQA(nn.Module):
    def __init__(self, num_frozen_layer, frozen_emb, frozen_layer_list):
        super().__init__()
        self.num_frozen_layer = num_frozen_layer
        self.embedding = frozen_emb
        self.layers = nn.ModuleList()
        logging.info("len(self.frozen_layer_list) = %d" % len(frozen_layer_list))
        for layer_i in range(num_frozen_layer * 2):
            self.layers.append(frozen_layer_list[layer_i])
        logging.info("len(self.layers) = %d" % len(self.layers))

    def forward(self, x, layer_id=0):
        if layer_id == self.num_frozen_layer:
            logging.info("no need to recompute")
            return x
        if layer_id == 0:
            logging.info("compute from layer 0")
            x = self.embedding(x)
            for id in range(0, self.num_frozen_layer * 2):
                x = self.layers[id](x)
            return x
        else:
            logging.info("compute from layer %d-%d" % (layer_id, self.num_frozen_layer - 1))
            for id in range(layer_id * 2, self.num_frozen_layer * 2):
                x = self.layers[id](x)
            return x


def create_pipe_styled_model_BERT_for_QA(model_config, model_backbone, num_layer_in_total,
                                         num_frozen_layer):
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
        size_embedding = count_parameters(frozen_emb, False)
        parameters_size_frozen += size_embedding

        frozen_model_sequential = nn.Sequential()
        # add transformer blocks needed to be trained
        for frozen_layer_index in range(num_frozen_layer):
            layer_block = model_backbone.bert.encoder.layer[frozen_layer_index]
            for param in layer_block.parameters():
                param.requires_grad = False
            size_layer_block = count_parameters(layer_block, False)
            parameters_size_frozen += size_layer_block

            # each layer has two sub layers: attention and FFN
            frozen_model_sequential.add_module("layer" + str(frozen_layer_index) + "attention", layer_block.attention)
            ffn_layer = BertFFNLayerForQA(model_config, layer_block.intermediate, layer_block.output)
            frozen_model_sequential.add_module("layer" + str(frozen_layer_index) + "ffn_layer", ffn_layer)

        frozen_model = BertFrozenLayerForQA(num_frozen_layer, frozen_emb, frozen_model_sequential)
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

        ffn_layer = BertFFNLayerForQA(model_config, layer_block.intermediate, layer_block.output)

        pipe_model.add_module("layer" + str(layer_index) + "ffn_layer", ffn_layer)
        size_layer_ffn_layer = count_parameters(ffn_layer, False)
        parameters_list_pipe.append(size_layer_ffn_layer)
        # logging.info(size_layer_ffn_layer)

    # QA model does not has the "pooler" layer
    # pipe_model.add_module("pooler", model_backbone.bert.pooler)
    # size_pooler = count_parameters(model_backbone.bert.pooler, False)
    # parameters_list_pipe.append(size_pooler)

    output_head = BertForQA_OutputHead(model_config)
    pipe_model.add_module("output_head", output_head)
    size_output_head = count_parameters(output_head, False)
    parameters_list_pipe.append(size_output_head)

    logging.info(frozen_model)
    logging.info(parameters_size_frozen)
    logging.info(pipe_model)
    logging.info(parameters_list_pipe)

    return frozen_model, parameters_size_frozen, pipe_model, parameters_list_pipe
