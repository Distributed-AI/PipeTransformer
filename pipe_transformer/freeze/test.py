import math


def freeze_algorithm(layer_num, alpha, epoch):
    second_term = 0.0
    for e in range(2, epoch+1):
        second_term += ((layer_num*alpha) / pow(1-alpha, e))
    return pow(1-alpha, epoch) * ((layer_num*alpha)/(1-alpha) + second_term)


if __name__ == "__main__":
    layer_num = 12
    alpha = 0.5
    frozen_layer_num_dict = dict()
    frozen_layer_num_dict[0] = 0
    for e in range(1, 10):
        frozen_layer_num_dict[e] = math.ceil(freeze_algorithm(layer_num, alpha, e))
    print(frozen_layer_num_dict)
