import logging
import math

import numpy as np


def generate_parameter_size_wise_balance(num_devices, param_list, num_frozen_layer):
    # B = {b0, ... b11}
    # b0 is real number
    # partition B into N partitions (pipe len)
    #
    balanced_layer_num = {}  # key: device_id; value: layer_num
    balanced_params_size = {}  # key: device_id; value: params_size

    assigned_layer_cnt = 0
    for i in range(num_devices):
        balanced_layer_num[i] = 0
        balanced_params_size[i] = 0.0
    # assign all frozen layers to the 1st device:
    frozen_layer_cost_factor = 1.0 / 6.0
    frozen_params = 0.0

    if num_frozen_layer > 0:
        # +1 because we need to include the 1st embedding layer
        # *2 because every transformer block contains two sub layers
        end_layer = range(num_frozen_layer * 2 + 1)
        for layer_idx in end_layer:
            # # make sure the last device has FOUR sub layers at least
            # if len(param_list) - layer_idx <= 4:
            #     break
            p = param_list[layer_idx]
            balanced_layer_num[0] += 1
            balanced_params_size[0] += p
            assigned_layer_cnt += 1
            frozen_params += p
        logging.info("frozen_params = %f" % frozen_params)

    total_param_size_to_be_assigned = sum(param_list) - balanced_params_size[0]
    logging.info("total_param_size_to_be_assigned = %f" % total_param_size_to_be_assigned)

    # search a better partition
    for i in range(num_devices):
        mean = total_param_size_to_be_assigned / (num_devices - i)
        # logging.info("mean = %f" % mean)
        variance = np.var(param_list[assigned_layer_cnt:]) / (num_devices - i)
        # logging.info("variance = %f" % variance)

        for idx in range(assigned_layer_cnt, len(param_list)):
            p = param_list[idx]
            if i == 0:
                criterion = balanced_params_size[i] - frozen_params * (1.0 - frozen_layer_cost_factor) + p
            else:
                criterion = balanced_params_size[i] + p
            if i == num_devices - 1 or criterion < mean + variance:
                balanced_params_size[i] += p
                balanced_layer_num[i] += 1
                assigned_layer_cnt += 1
                total_param_size_to_be_assigned -= p
            else:
                break

    # evaluate the theory of "Block Partitions of Sequences"
    check_gap_all(balanced_params_size, frozen_params, frozen_layer_cost_factor)
    check_gap_except_1st_layer(balanced_params_size)

    return balanced_layer_num, balanced_params_size, frozen_params


def check_gap_all(balanced_params_size, frozen_params, frozen_layer_cost_factor):
    list_except_frozen_layers = []
    for k in balanced_params_size.keys():
        if k == 0:
            list_except_frozen_layers.append(balanced_params_size[0] - frozen_params +
                                             frozen_params * frozen_layer_cost_factor)
        else:
            list_except_frozen_layers.append(balanced_params_size[k])

    max_params_in_balanced_partition = max(list_except_frozen_layers)
    min_params_in_balanced_partition = min(list_except_frozen_layers)
    gap = abs(max_params_in_balanced_partition - min_params_in_balanced_partition)
    logging.info("max_params_in_balanced_partition - min_params_in_balanced_partition = %f" % gap)
    # if gap > min_params_in_balanced_partition:
    #   raise Exception("Error: the partition should meet the theory of 'Block Partitions of Sequences'")


def check_gap_except_1st_layer(balanced_params_size):
    list_except_frozen_layers = []
    for k in balanced_params_size.keys():
        if k != 0:
            list_except_frozen_layers.append(balanced_params_size[k])

    if len(list_except_frozen_layers) == 0:
        return
    max_params_in_balanced_partition = max(list_except_frozen_layers)
    min_params_in_balanced_partition = min(list_except_frozen_layers)
    gap = abs(max_params_in_balanced_partition - min_params_in_balanced_partition)
    logging.info("max_params_in_balanced_partition - min_params_in_balanced_partition = %f" % gap)
    # if gap > min_params_in_balanced_partition:
    #     raise Exception("Error: the gap is too big!")


def generate_layer_wise_balance(num_devices, num_layers):
    balance = []
    layers_assigned = 0
    for i in range(num_devices):
        x = (num_layers - layers_assigned) / (num_devices - i)
        if x.is_integer():
            balance.append(int(x))
            layers_assigned += x
        else:
            balance.append(math.ceil(x))
            layers_assigned += math.ceil(x)
    return balance
