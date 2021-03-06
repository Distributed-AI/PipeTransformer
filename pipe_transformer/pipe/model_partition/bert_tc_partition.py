import logging

from torch import nn

from transformers import apply_chunking_to_forward
from .utils import count_parameters

"""
For BERT + Text Classification

Pipeline only support nn.Sequential partitions thus a layer cannot take more than 1 preceding layers as input.
This model architecture is normal in practice, such as Transformer Multi Head Attention layer, and the skip connection in ResNet.

Without such support, the partition can only be coarse-grained, which is hard to make the word load balanced in pipeline.

New insights:
When a model architecture contains skip connection, the partition is a trade-off between communication and computation. 
If we only consider the partition in the angle of parameter size, 
it will lead to extra cost in communication and caching when a skip connection is divided into two partitions (in different GPU devices), between which we have to move intermediate output from preceding partition across GPU device. 
If we make the communication balanced between any two partitions by maintaining the skip connection structure as an atomic operation, 
the caching will not have extra cost but the computational cost may not be balanced. We choose the second strategy because
in our experimental observation, we find that (1) the hidden feature map for a batch (batch size 400) is extremely large in BERT and Vision Transformers (more than 768*512*400*4*8/1e6
5 Gb), this requires too much communication cost (2) when making the communication equal between any two partitions, 
the gap of parameter size between two partition is only around 2.5M, and the time cost gap is far more less than the communication cost of a large batch of hidden feature maps.
Based on the strategy of maintaining skip connection in Transformers as an atomic operation, we design a load balanced algorithm, which aims to make the partition balanced in terms of parameter size.
Furthermore, with the profiling in a real hardware environment, we dynamically switch the optimal number of micro-batches in the pipeline after the pipeline has transformed (when the length has been changed). 
This further mitigates the parameter size gap in some degree. 
All in all, we believe this design is the optimal strategy in our PipeTransformer training system. 
"""


# class BertIntermediateLayerForTC(nn.Module):
#     def __init__(self, config, intermediate):
#         super().__init__()
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward
#         self.seq_len_dim = 1
#         self.intermediate = intermediate
#
#     def forward(self, self_attention_outputs):
#         attention_output = self_attention_outputs[0]
#         # outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
#         # layer_output = apply_chunking_to_forward(
#         #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
#         # )
#         intermediate_output = self.intermediate(attention_output)
#         return intermediate_output, attention_output
#
#
# class BertOutputLayerForTC(nn.Module):
#     def __init__(self, config, output):
#         super().__init__()
#         self.output = output
#
#     def forward(self, intermediate_output, attention_output):
#         return self.output(intermediate_output, attention_output)


class BertFFNLayerForTC(nn.Module):
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


class BertForSequenceClassification_OutputHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, bert_outputs):
        pooled_output = self.dropout(bert_outputs)
        logits = self.classifier(pooled_output)
        return logits


class BertFrozenLayer(nn.Module):
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


def create_pipe_styled_model_BERT_for_TC(model_config, model_backbone, num_layer_in_total,
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

            ffn_layer = BertFFNLayerForTC(model_config, layer_block.intermediate, layer_block.output)
            frozen_model_sequential.add_module("layer" + str(frozen_layer_index) + "ffn_layer",
                                               ffn_layer)

        frozen_model = BertFrozenLayer(num_frozen_layer, frozen_emb, frozen_model_sequential)
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

        ffn_layer = BertFFNLayerForTC(model_config, layer_block.intermediate, layer_block.output)
        pipe_model.add_module("layer" + str(layer_index) + "ffn_layer", ffn_layer)
        size_layer_intermediate_layer = count_parameters(ffn_layer, False)
        parameters_list_pipe.append(size_layer_intermediate_layer)
        # logging.info(size_layer_ffn_layer)

    pipe_model.add_module("pooler", model_backbone.bert.pooler)
    size_pooler = count_parameters(model_backbone.bert.pooler, False)
    parameters_list_pipe.append(size_pooler)

    output_head = BertForSequenceClassification_OutputHead(model_config)
    pipe_model.add_module("output_head", output_head)
    size_output_head = count_parameters(output_head, False)
    parameters_list_pipe.append(size_output_head)

    logging.info(frozen_model)
    logging.info(parameters_size_frozen)
    logging.info(pipe_model)
    logging.info(parameters_list_pipe)

    return frozen_model, parameters_size_frozen, pipe_model, parameters_list_pipe
