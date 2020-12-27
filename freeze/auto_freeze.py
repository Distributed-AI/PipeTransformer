class AutoFreeze:
    def __init__(self):
        self.num_freeze_layers = 0
        self.is_freeze = True

    def do_freeze(self):
        self.is_freeze = True

    def do_not_freeze(self):
        self.is_freeze = False

    def is_freeze_open(self):
        return self.is_freeze

    def get_hand_crafted_frozen_layers_by_epoch(self, epoch):
        num_freeze_layers = 0
        if epoch == 0:
            num_freeze_layers = 0
        elif epoch > 1 and epoch <= 2:
            num_freeze_layers = 4
        elif epoch > 3 and epoch <= 5:
            num_freeze_layers = 8
        elif epoch > 5 and epoch <= 7:
            num_freeze_layers = 10
        elif epoch > 7:
            num_freeze_layers = 12
        return num_freeze_layers
