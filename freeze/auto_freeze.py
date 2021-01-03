import logging

import numpy as np
import torch


class AutoFreeze:
    def __init__(self):
        self.model = None
        self.num_freeze_layers = 0
        self.is_freeze = True

        self.prev_intermediate_grad_dict = None
        self.grad_eval_iteration = 500
        self.percentile = 50

    def do_freeze(self):
        self.is_freeze = True

    def do_not_freeze(self):
        self.is_freeze = False

    def is_freeze_open(self):
        return self.is_freeze

    def get_hand_crafted_frozen_layers_by_epoch(self, epoch):
        num_freeze_layers = 0
        if epoch == 0:
            num_freeze_layers = 6
        elif epoch > 1 and epoch <= 2:
            num_freeze_layers = 6
        elif epoch > 2 and epoch <= 5:
            num_freeze_layers = 8
        elif epoch > 5 and epoch <= 7:
            num_freeze_layers = 10
        elif epoch > 7:
            num_freeze_layers = 12
        return num_freeze_layers

    def start_accumulate(self, model):
        self.grad_tensor_dict = {}
        for name, param in model.named_parameters():
            self.grad_tensor_dict[name] = torch.zeros(param.shape).to(0)

    def accumulate(self, model):
        for name, param in model.bert.named_parameters():
            if param.grad is not None:
                if name not in self.grad_tensor_dict.keys():
                    self.grad_tensor_dict[name] = param.grad
                else:
                    self.grad_tensor_dict[name] += param.grad

    def freeze(self, model, device):
        current_grad_dict = {}
        for i in range(12):
            current_grad_dict[i] = 0
        for name in self.grad_tensor_dict.keys():
            param_list = name.split(".")
            layer_num = 0
            for split_param in param_list:
                try:
                    layer_num = int(split_param)
                    if "encoder" in name:
                        current_grad_dict[layer_num] += torch.norm(self.grad_tensor_dict[name].cpu().detach(),
                                                                   p=1).item()
                except ValueError:
                    pass

        # Clear gradient accumulator
        for name, param in self.model.bert.named_parameters():
            self.grad_tensor_dict[name] = torch.zeros(param.shape).to(device)

        if self.prev_intermediate_grad_dict is None:
            # Set gradient dict to be compared with for the first time
            self.prev_intermediate_grad_dict = current_grad_dict
        else:
            threshold_dict = {}
            for key in range(12):
                threshold_dict[key] = 0
            # Calculate gradient changing threshold
            for key in current_grad_dict.keys():
                if current_grad_dict[key] > 0:
                    threshold_dict[key] = abs(self.prev_intermediate_grad_dict[key] - current_grad_dict[key]) / \
                                          self.prev_intermediate_grad_dict[key]

            median_value = np.percentile(list(threshold_dict.values())[self.num_freeze_layers:], self.percentile)
            # Find out the first layer with ratio ge to the median value
            for key in threshold_dict.keys():
                if threshold_dict[key] >= median_value:
                    start_layer = key
                    break
            self.prev_intermediate_grad_dict = current_grad_dict
            logging.info("threshold: ", threshold_dict)
            logging.info("layer num: ", self.num_freeze_layers)


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)


if __name__ == "__main__":
    auto_freeze = AutoFreeze()
