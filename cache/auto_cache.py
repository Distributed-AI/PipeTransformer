class AutoCache:
    def __init__(self):
        self.num_frozen_layers = 0
        self.train_extracted_features = dict()
        self.test_extracted_features = dict()

    def update_num_frozen_layers(self, num_frozen_layers):
        self.num_frozen_layers = num_frozen_layers
        self.train_extracted_features.clear()
        self.test_extracted_features.clear()

    def cache_train_extracted_hidden_feature(self, batch_idx, extracted_feature):
        self.train_extracted_features[batch_idx] = extracted_feature.cpu().pin_memory()

    def cache_test_extracted_hidden_feature(self, batch_idx, extracted_feature):
        self.test_extracted_features[batch_idx] = extracted_feature.cpu().pin_memory()

    def get_train_extracted_hidden_feature(self, batch_idx):
        # the hidden features are always in device 0
        return self.train_extracted_features[batch_idx].to("cuda:0")

    def get_test_extracted_hidden_feature(self, batch_idx):
        # the hidden features are always in device 0
        return self.test_extracted_features[batch_idx].to("cuda:0")
