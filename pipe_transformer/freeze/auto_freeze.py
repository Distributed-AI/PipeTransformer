import copy
import logging
import math

import numpy as np
import torch

from .shared_memory_manager_int_value import SharedMemoryManagerIntValue


class AutoFreeze:
    def __init__(self, config):
        self.model = None
        self.num_freeze_layers = 0
        self.is_freeze = config.b_freeze
        self.is_hand_crafted = True

        self.is_grad_norm_analysis = False

        self.num_layer = 12
        self.grad_accumulated_by_layer = dict()
        for layer_idx in range(self.num_layer):
            self.grad_accumulated_by_layer[layer_idx] = dict()
        self.is_grad_accumulated_by_layer_updated = False

        self.freeze_interval = 1

        self.last_grad_norm_by_layer = None
        self.percentile = 50

        self.shared_memory_mgr_frozen_layer_num = SharedMemoryManagerIntValue("frozen_layer_num")

        self.freeze_strategy = "linear"

        self.frozen_layer_linear = dict()
        epochs = 10
        for e in range(epochs):
            progress = math.ceil(((e+1)/epochs)*self.num_layer)
            self.frozen_layer_linear[e] = int(progress)

    def update_status(self, num_freeze_layers, last_grad_norm_by_layer):
        logging.info("(%s) num_freeze_layers = %d, last_grad_norm_by_layer = %s" % (
            str(id(self)), num_freeze_layers, str(last_grad_norm_by_layer)))
        self.num_freeze_layers = num_freeze_layers
        if last_grad_norm_by_layer is not None:
            self.last_grad_norm_by_layer = copy.deepcopy(last_grad_norm_by_layer)

    def get_status(self):
        return self.num_freeze_layers, self.last_grad_norm_by_layer

    def enable(self, on):
        self.is_freeze = on

    def is_freeze_open(self):
        return self.is_freeze

    def cleanup(self):
        self.shared_memory_mgr_frozen_layer_num.cleanup()
        pass

    # def get_hand_crafted_frozen_layers_by_epoch(self, epoch):
    #     num_freeze_layers = 0
    #     if not self.shared_memory_mgr_frozen_layer_num.is_exist(epoch):
    #         self.shared_memory_mgr_frozen_layer_num.add_int_value(epoch, num_freeze_layers)
    #     return num_freeze_layers

    # def get_hand_crafted_frozen_layers_by_epoch(self, epoch):
    #     num_freeze_layers = 6
    #     if not self.shared_memory_mgr_frozen_layer_num.is_exist(epoch):
    #         self.shared_memory_mgr_frozen_layer_num.add_int_value(epoch, num_freeze_layers)
    #     return num_freeze_layers

    # def get_hand_crafted_frozen_layers_by_epoch(self, epoch):
    #     num_freeze_layers = epoch
    #     if not self.shared_memory_mgr_frozen_layer_num.is_exist(epoch):
    #         self.shared_memory_mgr_frozen_layer_num.add_int_value(epoch, num_freeze_layers)
    #     return num_freeze_layers

    def get_hand_crafted_frozen_layers_by_epoch(self, epoch):
        if self.freeze_strategy == "linear":
            num_freeze_layers = self.frozen_layer_linear[epoch]
        elif self.freeze_strategy == "mild":
            num_freeze_layers = 0
            if epoch == 0:
                num_freeze_layers = 0
            elif epoch >= 1 and epoch <= 2:
                num_freeze_layers = 6
            elif epoch > 2 and epoch <= 5:
                num_freeze_layers = 8
            elif epoch > 5 and epoch <= 7:
                num_freeze_layers = 10
            elif epoch > 7:
                num_freeze_layers = 12
        elif self.freeze_strategy == "start_from_freeze_all":
            num_freeze_layers = 12
        elif self.freeze_strategy == "pipe_length_4":
            num_freeze_layers = 6
        elif self.freeze_strategy == "pipe_length_2":
            num_freeze_layers = 8
        elif self.freeze_strategy == "freeze_by_epoch":
            num_freeze_layers = epoch
        else:
            raise Exception("no such strategy")
        if not self.shared_memory_mgr_frozen_layer_num.is_exist(epoch):
            self.shared_memory_mgr_frozen_layer_num.add_int_value(epoch, num_freeze_layers)
        return num_freeze_layers

    def get_num_of_frozen_layer(self, epoch):
        return self.shared_memory_mgr_frozen_layer_num.get_int_value(epoch)

    def accumulate(self, model):
        for layer_idx in range(self.num_layer):
            for name, param in model.transformer.encoder.layer[layer_idx].named_parameters():
                if param.grad is not None:
                    if name not in self.grad_accumulated_by_layer.keys():
                        self.grad_accumulated_by_layer[layer_idx][name] = param.grad
                    else:
                        self.grad_accumulated_by_layer[layer_idx][name] += param.grad
                    # logging.info("layer_idx = %d, name = %s" % (layer_idx, name))
        self.is_grad_accumulated_by_layer_updated = True

    def freeze(self, epoch):
        logging.info("-----------------------------%s" % (id(self)))
        if self.is_hand_crafted:
            return self.get_hand_crafted_frozen_layers_by_epoch(epoch)
        if epoch == 0:
            return 0

        if not self.is_grad_accumulated_by_layer_updated:
            return self.num_freeze_layers
        if self.num_freeze_layers == self.num_layer:
            return self.num_freeze_layers

        if (epoch + 1) % self.freeze_interval == 0:
            # Calculate layer-wise gradient changing ratio
            grad_norm_by_layer = dict()
            for i in range(self.num_layer):
                grad_norm_by_layer[i] = 0

            for layer_idx in self.grad_accumulated_by_layer.keys():
                for name in self.grad_accumulated_by_layer[layer_idx].keys():
                    grad = self.grad_accumulated_by_layer[layer_idx][name]
                    grad_norm_by_layer[layer_idx] += torch.norm(grad.cpu().detach(), p=1).item()

            # Clear gradient accumulator
            for layer_idx in self.grad_accumulated_by_layer.keys():
                for name in self.grad_accumulated_by_layer[layer_idx].keys():
                    params = self.grad_accumulated_by_layer[layer_idx][name]
                    self.grad_accumulated_by_layer[layer_idx][name] = torch.zeros(params.shape)
            self.is_grad_accumulated_by_layer_updated = False

            logging.info("epoch = %d, grad_norm_by_layer = %s" % (epoch, str(grad_norm_by_layer)))
            frozen_layer_idx = -1
            if self.last_grad_norm_by_layer is None:
                # Set gradient dict to be compared with for the first time
                self.last_grad_norm_by_layer = grad_norm_by_layer
            else:
                grad_norm_diff = dict()
                # Calculate gradient changing threshold
                for key in grad_norm_by_layer.keys():
                    if grad_norm_by_layer[key] > 0:
                        grad_norm_diff[key] = abs(self.last_grad_norm_by_layer[key] - grad_norm_by_layer[key]) / \
                                              self.last_grad_norm_by_layer[key]
                    else:
                        grad_norm_diff[key] = 0

                logging.info(grad_norm_diff)
                unfrozen_list = list(grad_norm_diff.values())[self.num_freeze_layers:]
                logging.info(unfrozen_list)
                logging.info("epoch = %d, grad_norm_diff (unfrozen_list) = %s" % (epoch, str(unfrozen_list)))
                grad_norm_diff_percentile = np.percentile(unfrozen_list, self.percentile)
                logging.info("grad_norm_diff_percentile = " + str(grad_norm_diff_percentile))

                # Find out the first layer with ratio ge to the median value
                for layer_idx in grad_norm_diff.keys():
                    if grad_norm_diff[layer_idx] >= grad_norm_diff_percentile:
                        frozen_layer_idx = layer_idx
                        break

                self.last_grad_norm_by_layer = grad_norm_by_layer
            logging.info("epoch = %d, frozen_layer_idx = %s" % (epoch, str(frozen_layer_idx)))
            # only analyze the grad norm
            if self.is_grad_norm_analysis:
                return 0
            if frozen_layer_idx != -1:
                self.num_freeze_layers = frozen_layer_idx + 1
                self.shared_memory_mgr_frozen_layer_num.add_int_value(epoch, self.num_freeze_layers)
        logging.info("epoch = %d, num_frozen_layer = %s" % (epoch, str(self.num_freeze_layers)))
        return self.num_freeze_layers
