class AutoCache:
    def __init__(self):
        self.num_frozen_layers = 0
        self.train_extracted_features = dict()
        self.test_extracted_features = dict()

        self.is_enable = False

    def update_num_frozen_layers(self, num_frozen_layers):
        self.num_frozen_layers = num_frozen_layers
        self.train_extracted_features.clear()
        self.test_extracted_features.clear()

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def infer_train(self, model, x, batch_idx):
        if self.is_enable:
            if self.get_train_extracted_hidden_feature(batch_idx) is None:
                hidden_feature = model.frozen_layers(x)
                self.cache_train_extracted_hidden_feature(batch_idx, hidden_feature)
            else:
                hidden_feature = self.get_train_extracted_hidden_feature(batch_idx)
            log_probs = model.active_layers(hidden_feature)
        else:
            log_probs = model(x)
        return log_probs

    def infer_test(self, model, x, batch_idx):
        if self.is_enable:
            if self.get_test_extracted_hidden_feature(batch_idx) is None:
                hidden_feature = model.frozen_layers(x)
                self.cache_test_extracted_hidden_feature(batch_idx, hidden_feature)
            else:
                hidden_feature = self.get_test_extracted_hidden_feature(batch_idx)
            log_probs = model.active_layers(hidden_feature)
        else:
            log_probs = model(x)
        return log_probs

    def cache_train_extracted_hidden_feature(self, batch_idx, extracted_feature):
        if not self.is_enable:
            return
        self.train_extracted_features[batch_idx] = extracted_feature.cpu().pin_memory()

    def cache_test_extracted_hidden_feature(self, batch_idx, extracted_feature):
        if not self.is_enable:
            return
        self.test_extracted_features[batch_idx] = extracted_feature.cpu().pin_memory()

    def get_train_extracted_hidden_feature(self, batch_idx):
        if not self.is_enable:
            return None
        # the hidden features are always in device 0
        return self.train_extracted_features[batch_idx].to("cuda:0")

    def get_test_extracted_hidden_feature(self, batch_idx):
        if not self.is_enable:
            return None
        # the hidden features are always in device 0
        return self.test_extracted_features[batch_idx].to("cuda:0")
