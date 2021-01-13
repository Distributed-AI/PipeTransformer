def get_hand_crafted_frozen_layers_by_epoch(self, epoch):
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
    self.shared_memory_dict_frozen_layer_num[epoch] = num_freeze_layers
    return num_freeze_layers

shared_memory_dict_frozen_layer_num = dict()

epoch = 8
current_num_frozen_layers = shared_memory_dict_frozen_layer_num[epoch]
while epoch >= 0:
    last_num_frozen_layers = self.shared_memory_dict_frozen_layer_num[epoch]
    if last_num_frozen_layers != current_num_frozen_layers:
        break
    epoch -= 1
print(current_num_frozen_layers)