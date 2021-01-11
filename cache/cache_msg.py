
class Message(object):
    MSG_TYPE_UPDATE_INDEX = 1
    MSG_TYPE_TRAINING_PROGRESS = 2
    MSG_TYPE_RESET = 3

    MSG_KEY_EPOCH = "epoch"
    MSG_KEY_BATCH_INDEX = "batch_idx"
    MSG_KEY_TRAIN_SAMPLE_INDEX = "train_sample_index"
    MSG_KEY_TEST_SAMPLE_INDEX = "test_sample_index"

    def __init__(self, msg_type):
        self.msg_type = msg_type
        self.msg_params = dict()

    def get_type(self):
        return self.msg_type

    def set(self, key, value):
        self.msg_params[key] = value

    def get(self, key):
        return self.msg_params[key]

    def get_params(self):
        return self.msg_params